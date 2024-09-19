from __future__ import annotations
from typing import TYPE_CHECKING, cast, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike

from scipy.optimize import minimize

from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV2 as Estimator, PrimitiveResult
from qiskit.primitives.containers.observables_array import (
    ObservablesArray,
    ObservablesArrayLike,
)
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.pub_result import PubResult
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import (
    QAOAAnsatz,
    TwoLocal as TwoLocalAnsatz,
    EfficientSU2 as EfficientSU2Ansatz,
)

from q3as.optimizer import Optimizer, COBYLA
from q3as.estimator import EstimatorOptions
import q3as.encoding
import q3as.api

if TYPE_CHECKING:
    from q3as.api import ApiClient, MaybeAwaitable, JobInfo


class _EvaluationData:
    evs: float
    stds: float


@dataclass
class VQEIteration:
    iter: int
    cost: float
    params: np.ndarray
    result: PrimitiveResult[PubResult]
    best: bool = False


class HaltReason(Enum):
    TOLERANCE = "tolerance"
    INTERRUPT = "interrupt"
    MAXITER = "maxiter"


@dataclass
class VQEResult:
    iter: int = 0
    reason: HaltReason = HaltReason.INTERRUPT
    best: Optional[VQEIteration] = None


@dataclass
class VQE:
    ansatz: QuantumCircuit
    observables: ObservablesArray
    initial_params: np.ndarray
    optimizer: Optimizer
    maxiter: Optional[int] = None

    @classmethod
    def builder(cls) -> VQEBuilder:
        return VQEBuilder()

    def run(
        self,
        estimator: Estimator,
        callback: Optional[Callable[[VQEIteration], None]] = None,
    ) -> VQEResult:
        out = VQEResult()

        def cost_fun(
            params: np.ndarray,
        ):
            out.iter += 1
            bound_params = BindingsArray.coerce({tuple(self.ansatz.parameters): params})
            pub = EstimatorPub(self.ansatz, self.observables, bound_params)
            result = estimator.run([pub]).result()
            cost = cast(_EvaluationData, result[0].data).evs
            vqe_result = VQEIteration(
                iter=out.iter, cost=cost, params=params, result=result
            )
            if out.best is None or cost < out.best.cost:
                vqe_result.best = True
                out.best = vqe_result
            if callback is not None:
                callback(vqe_result)
            return cost

        options = {}
        if self.maxiter is not None:
            options["maxiter"] = self.maxiter

        try:
            res = minimize(
                cost_fun,
                self.initial_params,
                method=self.optimizer.scipy_method,
                options=options,
            )
            if res.success:
                out.reason = HaltReason.TOLERANCE
            else:
                out.reason = HaltReason.MAXITER
        except StopIteration:
            out.reason = HaltReason.INTERRUPT

        return out

    def send(
        self, api: ApiClient, estimator: EstimatorOptions = EstimatorOptions()
    ) -> MaybeAwaitable[JobInfo]:
        return api.create_job(
            q3as.api.JobRequest(
                input=q3as.encoding.EncodedVQE.encode(self), estimator=estimator
            )
        )


def _observables_to_pauli(observables: ObservablesArray) -> SparsePauliOp:
    return SparsePauliOp.from_list(observables.ravel().tolist()[0].items())


type _PostProcessCallable = Optional[Callable[[QuantumCircuit], QuantumCircuit]]


def _postprocessed(
    postprocess: _PostProcessCallable, ansatz: QuantumCircuit
) -> QuantumCircuit:
    if postprocess is not None:
        return postprocess(ansatz)
    return ansatz


def QAOA(
    postprocess: _PostProcessCallable = None, **kwargs
) -> Callable[[ObservablesArray], QuantumCircuit]:
    return lambda obs: _postprocessed(
        postprocess, QAOAAnsatz(_observables_to_pauli(obs), **kwargs)
    )


def TwoLocal(
    postprocess: _PostProcessCallable = None, **kwargs
) -> Callable[[ObservablesArray], QuantumCircuit]:
    return lambda obs: _postprocessed(
        postprocess, TwoLocalAnsatz(_observables_to_pauli(obs).num_qubits, **kwargs)
    )


def EfficientSU2(
    postprocess: _PostProcessCallable = None, **kwargs
) -> Callable[[ObservablesArray], QuantumCircuit]:
    return lambda obs: _postprocessed(
        postprocess, EfficientSU2Ansatz(_observables_to_pauli(obs).num_qubits, **kwargs)
    )


class VQEBuilder:
    _ansatz: Optional[
        Union[QuantumCircuit, Callable[[ObservablesArray], QuantumCircuit]]
    ] = None
    _observables: Optional[ObservablesArrayLike] = None
    _initial_params: Optional[ArrayLike] = None
    _optimizer: Optimizer = COBYLA()
    _maxiter: Optional[int] = None

    def ansatz(
        self, qc: Union[QuantumCircuit, Callable[[ObservablesArray], QuantumCircuit]]
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
            raise ValueError("Observables are required")
        observables = ObservablesArray.coerce(self._observables)

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
            ansatz=ansatz,
            observables=observables,
            initial_params=_initial_params,
            optimizer=self._optimizer,
            maxiter=self._maxiter,
        )

    def run(
        self,
        estimator: Estimator,
        callback: Optional[Callable[[VQEIteration], None]] = None,
    ) -> VQEResult:
        return self.build().run(estimator, callback)

    def send(
        self, api: ApiClient, estimator: EstimatorOptions = EstimatorOptions()
    ) -> MaybeAwaitable[JobInfo]:
        return self.build().send(api, estimator)
