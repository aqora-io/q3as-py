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
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.sampler_pub_result import SamplerPubResult
from qiskit.primitives.containers.pub_result import PubResult
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import (
    QAOAAnsatz,
    TwoLocal as TwoLocalAnsatz,
    EfficientSU2 as EfficientSU2Ansatz,
)

from q3as.app import Application
from q3as.optimizer import Optimizer, COBYLA
from q3as.run_options import RunOptions
import q3as.encoding
import q3as.api

from .types import HaltReason, EstimatorData, SamplerData

if TYPE_CHECKING:
    from q3as.api import ApiClient, MaybeAwaitable, JobInfo


@dataclass
class VQEIteration:
    iter: int
    cost: float
    params: np.ndarray
    estimated: PrimitiveResult[PubResult]
    best: bool = False


@dataclass
class VQEResult:
    params: np.ndarray
    iter: int = 0
    reason: HaltReason = HaltReason.INTERRUPT
    cost: Optional[float] = None
    estimated: Optional[PrimitiveResult[PubResult]] = None
    sampled: Optional[PrimitiveResult[SamplerPubResult]] = None
    meas_counts: Optional[Dict[str, int]] = None
    interpreted: Optional[List[Tuple[Any, int]]] = None


@dataclass
class VQE:
    ansatz: QuantumCircuit
    observables: ObservablesArray
    initial_params: np.ndarray
    optimizer: Optimizer
    app: Optional[Application]
    maxiter: int = 1000

    @classmethod
    def builder(cls) -> VQEBuilder:
        return VQEBuilder()

    def run(
        self,
        estimator: Estimator,
        sampler: Optional[Sampler] = None,
        callback: Optional[Callable[[VQEIteration], None]] = None,
    ) -> VQEResult:
        out = VQEResult(params=self.initial_params)

        def cost_fun(
            params: np.ndarray,
        ):
            out.iter += 1
            bound_params = BindingsArray.coerce({tuple(self.ansatz.parameters): params})
            pub = EstimatorPub(self.ansatz, self.observables, bound_params)
            result = estimator.run([pub]).result()
            cost = cast(EstimatorData, result[0].data).evs
            vqe_result = VQEIteration(
                iter=out.iter, cost=cost, params=params, estimated=result
            )
            if out.cost is None or cost < out.cost:
                vqe_result.best = True
                out.params = params
                out.cost = cost
                out.estimated = vqe_result.estimated
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

            if self.app is not None:
                meas = cast(SamplerData, sampled[0].data).meas
                out.meas_counts = meas.get_counts()
                out.interpreted = self.app.interpreted_meas(meas)

        return out

    def send(
        self, api: ApiClient, run_options: RunOptions = RunOptions()
    ) -> MaybeAwaitable[JobInfo]:
        return api.create_job(
            q3as.api.JobRequest(
                input=q3as.encoding.EncodedVQE.encode(self), run_options=run_options
            )
        )


def _observables_to_pauli(observables: ObservablesArray) -> SparsePauliOp:
    return SparsePauliOp.from_list(observables.ravel().tolist()[0].items())


type _PostProcessCallable = Optional[Callable[[QuantumCircuit], QuantumCircuit]]


type AnsatzCallback = Callable[[ObservablesArray], QuantumCircuit]


class QAOA:
    def __init__(self, postprocess: _PostProcessCallable = None, **kwargs):
        self.postprocess = postprocess
        self.kwargs = kwargs

    def __call__(self, obs: ObservablesArray) -> QuantumCircuit:
        ansatz = QAOAAnsatz(_observables_to_pauli(obs), **self.kwargs)
        if self.postprocess is not None:
            ansatz = self.postprocess(ansatz)
        return ansatz


class TwoLocal:
    def __init__(self, postprocess: _PostProcessCallable = None, **kwargs):
        self.postprocess = postprocess
        self.kwargs = kwargs

    def __call__(self, obs: ObservablesArray) -> QuantumCircuit:
        ansatz = TwoLocalAnsatz(_observables_to_pauli(obs).num_qubits, **self.kwargs)
        if self.postprocess is not None:
            ansatz = self.postprocess(ansatz)
        return ansatz


class EfficientSU2:
    def __init__(self, postprocess: _PostProcessCallable = None, **kwargs):
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
        self._app = app
        return self

    def ansatz(
        self,
        qc: Union[QuantumCircuit, AnsatzCallback],
    ) -> VQEBuilder:
        self._ansatz = qc
        return self

    def observables(self, obs: ObservablesArrayLike) -> VQEBuilder:
        self._observables = obs
        return self

    def initial_params(self, params: ArrayLike) -> VQEBuilder:
        self._initial_params = params
        return self

    def optimizer(self, opt: Optimizer) -> VQEBuilder:
        self._optimizer = opt
        return self

    def maxiter(self, maxiter: int) -> VQEBuilder:
        self._maxiter = maxiter
        return self

    def build(self) -> VQE:
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
        return self.build().run(estimator, sampler, callback)

    def send(
        self, api: ApiClient, estimator: RunOptions = RunOptions()
    ) -> MaybeAwaitable[JobInfo]:
        return self.build().send(api, estimator)
