from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    cast,
    Optional,
    Union,
    Callable,
    Any,
    List,
    Tuple,
    Dict,
)
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from scipy.optimize import minimize

from qiskit.circuit import QuantumCircuit
from qiskit.primitives import (
    BaseEstimatorV2 as Estimator,
    BaseSamplerV2 as Sampler,
    PrimitiveResult,
)
from qiskit.primitives.containers.observables_array import (
    ObservablesArray,
    ObservablesArrayLike,
)
from qiskit.primitives.containers.bindings_array import BindingsArray, BindingsArrayLike
from qiskit.primitives.containers.sampler_pub_result import SamplerPubResult
from qiskit.primitives.containers.pub_result import PubResult
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import (
    QAOAAnsatz,
    TwoLocal as TwoLocalAnsatz,
    EfficientSU2 as EfficientSU2Ansatz,
)

from q3as.algo.cutting import (
    CutAction,
    create_estimate_cut,
)
from q3as.app import Application
from q3as.run_options import RunOptions
import q3as.encoding
import q3as.api

from .optimizer import Optimizer, COBYLA
from .types import HaltReason, SamplerData

if TYPE_CHECKING:
    from q3as.client import Client
    from q3as.api import Job


def _observables_to_pauli(observables: ObservablesArray) -> SparsePauliOp:
    return SparsePauliOp.from_list(observables.ravel().tolist()[0].items())


@dataclass
class VQEIteration:
    """
    The result of a an iteration of VQE
    """

    iter: int
    "The number of the iteration"
    cost: float
    "The value of the cost function of the iteration"
    params: np.ndarray
    "The parameters of the ansatz of the iteration"
    sampled: List[PrimitiveResult[SamplerPubResult]]
    "The result of the estimation"
    best: bool = False
    "Whether this iteration is the best so far"


@dataclass
class VQEResult:
    params: np.ndarray
    "The parameters of the best iteration"
    iter: int = 0
    "The number of iterations"
    reason: HaltReason = HaltReason.INTERRUPT
    "The reason why the optimization stopped"
    cost: Optional[float] = None
    "The value of the cost function of the best iteration"
    estimated: Optional[PrimitiveResult[PubResult]] = None
    "The result of the estimation of the best iteration"
    sampled: Optional[PrimitiveResult[SamplerPubResult]] = None
    "The result of the sampling"
    meas_counts: Optional[Dict[str, int]] = None
    "The bit string counts after sampling"
    interpreted: Optional[List[Tuple[Any, int]]] = None
    "The interpreted results of the sampling as defined by the application if given"

    def most_sampled(self) -> Optional[Any]:
        """
        Get the the most sampled value from the results.
        This is the interpreted value as defined by the application if given or otherwise the bitstring
        """
        samples = None
        if self.interpreted is not None:
            samples = [(count, value) for value, count in self.interpreted]
        elif self.meas_counts is not None:
            samples = [
                (count, bitstring) for bitstring, count in self.meas_counts.items()
            ]
        max_sample = max(samples, key=lambda x: x[0]) if samples is not None else None
        if max_sample is not None:
            return max_sample[1]
        return None


@dataclass
class VQE:
    """
    Parameters for running VQE
    """

    ansatz: QuantumCircuit
    "The ansatz circuit"
    observables: ObservablesArray
    "The observables to estimate. E.g. the Hamiltonian of the system"
    initial_params: np.ndarray
    "The initial parameters of the ansatz"
    optimizer: Optimizer
    "The optimizer to use"
    app: Optional[Application]
    "The application to use for interpretation of results"
    maxiter: int = 1000
    "The maximum number of iterations"

    @classmethod
    def builder(cls) -> VQEBuilder:
        """
        Create a builder for VQE
        """
        return VQEBuilder()

    def run(
        self,
        sampler: Sampler,
        *,
        cuts: List[CutAction] = [],
        num_cut_samples: int | float = np.inf,
        callback: Optional[Callable[[VQEIteration], None]] = None,
    ) -> VQEResult:
        """
        Run VQE
        """

        experiments = create_estimate_cut(
            self.ansatz,
            _observables_to_pauli(self.observables),
            cuts,
            num_cut_samples,
        )

        out = VQEResult(params=self.initial_params)

        def cost_fun(
            params: np.ndarray,
        ):
            out.iter += 1
            cost, sampled = experiments.run(sampler, params)
            vqe_result = VQEIteration(
                iter=out.iter, cost=cost, params=params, sampled=sampled
            )
            if out.cost is None or cost < out.cost:
                vqe_result.best = True
                out.params = params
                out.cost = cost
                out.sampled = vqe_result.sampled
            if callback is not None:
                callback(vqe_result)
            return cost

        try:
            res = minimize(
                cost_fun,
                self.initial_params,
                method=self.optimizer.scipy_method,
                options={"maxiter": self.maxiter},
            )
            if res.success:
                out.reason = HaltReason.TOLERANCE
            else:
                out.reason = HaltReason.MAXITER
        except StopIteration:
            out.reason = HaltReason.INTERRUPT

        if sampler is not None:
            bound_params = cast(
                BindingsArrayLike,
                BindingsArray.coerce({tuple(self.ansatz.parameters): out.params}).data,
            )
            measured_ansatz = cast(
                QuantumCircuit, self.ansatz.measure_all(inplace=False)
            )
            sampled = sampler.run([(measured_ansatz, bound_params)]).result()

            out.sampled = sampled

            meas = cast(SamplerData, sampled[0].data).meas
            out.meas_counts = meas.get_counts()

            if self.app is not None:
                out.interpreted = self.app.interpreted_meas(meas)

        return out

    def send(self, api: Client, run_options: RunOptions = RunOptions()) -> Job:
        """
        Send the VQE job to the API
        """
        return api.create_job(
            q3as.api.JobRequest(
                input=q3as.encoding.EncodedVQE.encode(self), run_options=run_options
            )
        )


type _PostProcessCallable = Optional[Callable[[QuantumCircuit], QuantumCircuit]]


type AnsatzCallback = Callable[[ObservablesArray], QuantumCircuit]


class QAOA:
    """
    A builder for the QAOA ansatz
    """

    def __init__(self, postprocess: _PostProcessCallable = None, **kwargs):
        """
        Create a new QAOA builder. Arguments are passed to the QAOAAnsatz constructor in the Qiskit circuit library.
        """
        self.postprocess = postprocess
        self.kwargs = kwargs

    def __call__(self, obs: ObservablesArray) -> QuantumCircuit:
        ansatz = QAOAAnsatz(_observables_to_pauli(obs), **self.kwargs)
        if self.postprocess is not None:
            ansatz = self.postprocess(ansatz)
        return ansatz


class TwoLocal:
    """
    A builder for the TwoLocal ansatz
    """

    def __init__(self, postprocess: _PostProcessCallable = None, **kwargs):
        """
        Create a new TwoLocal builder. Arguments are passed to the TwoLocal constructor in the Qiskit circuit library.
        """
        self.postprocess = postprocess
        self.kwargs = kwargs

    def __call__(self, obs: ObservablesArray) -> QuantumCircuit:
        ansatz = TwoLocalAnsatz(_observables_to_pauli(obs).num_qubits, **self.kwargs)
        if self.postprocess is not None:
            ansatz = self.postprocess(ansatz)
        return ansatz


class EfficientSU2:
    """
    A builder for the EfficientSU2 ansatz
    """

    def __init__(self, postprocess: _PostProcessCallable = None, **kwargs):
        """
        Create a new EfficientSU2 builder. Arguments are passed to the EfficientSU2 constructor in the Qiskit circuit library.
        """
        self.postprocess = postprocess
        self.kwargs = kwargs

    def __call__(self, obs: ObservablesArray) -> QuantumCircuit:
        ansatz = EfficientSU2Ansatz(
            _observables_to_pauli(obs).num_qubits, **self.kwargs
        )
        if self.postprocess is not None:
            ansatz = self.postprocess(ansatz)
        return ansatz


class VQEBuilder:
    _app: Optional[Application] = None
    _ansatz: Optional[Union[QuantumCircuit, AnsatzCallback]] = EfficientSU2()
    _observables: Optional[ObservablesArrayLike] = None
    _initial_params: Optional[ArrayLike] = None
    _optimizer: Optimizer = COBYLA()
    _maxiter: int = 1000

    def app(self, app: Application):
        """
        Set the `Application` to use for interpretation of results
        """
        self._app = app
        return self

    def ansatz(
        self,
        qc: Union[QuantumCircuit, AnsatzCallback],
    ) -> VQEBuilder:
        """
        Set the ansatz to use for the VQE.
        This can either be a QuantumCircuit or a callback that takes the observables as argument and returns a QuantumCircuit.
        This defaults to the EfficientSU2 ansatz.
        """
        self._ansatz = qc
        return self

    def observables(self, obs: ObservablesArrayLike) -> VQEBuilder:
        """
        Set the observables to estimate. E.g. the Hamiltonian of the system.
        If no observables are given, the Hamiltonian of the application is used
        """
        self._observables = obs
        return self

    def initial_params(self, params: ArrayLike) -> VQEBuilder:
        """
        Set the initial parameters of the ansatz. If no parameters are given, the initial parameters are set to 0.
        """
        self._initial_params = params
        return self

    def optimizer(self, opt: Optimizer) -> VQEBuilder:
        """
        Set the optimizer to use for the VQE. This defaults to COBYLA.
        """
        self._optimizer = opt
        return self

    def maxiter(self, maxiter: int) -> VQEBuilder:
        """
        Set the maximum number of iterations for the VQE. This defaults to 1000.
        """
        self._maxiter = maxiter
        return self

    def build(self) -> VQE:
        """
        Build the VQE instance
        """
        if self._observables is None:
            if self._app is not None:
                observables = self._app.hamiltonian()
            else:
                raise ValueError("Observables are required")
        else:
            observables = self._observables
        observables = ObservablesArray.coerce(observables)

        if self._ansatz is None:
            raise ValueError("Ansatz is required")
        if isinstance(self._ansatz, QuantumCircuit):
            ansatz = self._ansatz
        else:
            ansatz = self._ansatz(observables)

        if self._initial_params is not None:
            _initial_params = np.array(self._initial_params)
            if len(_initial_params) != ansatz.num_parameters:
                raise ValueError("Initial parameters must match ansatz")
        else:
            _initial_params = np.zeros(ansatz.num_parameters)

        return VQE(
            app=self._app,
            ansatz=ansatz,
            observables=observables,
            initial_params=_initial_params,
            optimizer=self._optimizer,
            maxiter=self._maxiter,
        )

    def run(
        self,
        estimator: Estimator,
        sampler: Optional[Sampler] = None,
        callback: Optional[Callable[[VQEIteration], None]] = None,
    ) -> VQEResult:
        """
        Run the VQE
        """
        return self.build().run(estimator, sampler, callback)

    def send(self, api: Client, estimator: RunOptions = RunOptions()) -> Job:
        """
        Send the VQE job to the API
        """
        return self.build().send(api, estimator)
