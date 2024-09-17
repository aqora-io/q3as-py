from __future__ import annotations
from collections.abc import Callable
from typing import cast, Optional, TYPE_CHECKING
from dataclasses import dataclass

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

from q3as.optimizer import Optimizer, COBYLA
from q3as.estimator import EstimatorOptions
import q3as.encoding
import q3as.api

if TYPE_CHECKING:
    from q3as.api import ApiClient, MaybeAwaitable, JobInfo


class _EvaluationData:
    evs: float


@dataclass
class VQEIteration:
    iter: int
    cost: float
    params: np.ndarray
    result: PrimitiveResult[PubResult]


@dataclass
class VQEResult:
    iter: int = 0
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
            if self.maxiter is not None:
                if out.iter >= self.maxiter:
                    raise StopIteration
            bound_params = BindingsArray.coerce({tuple(self.ansatz.parameters): params})
            pub = EstimatorPub(self.ansatz, self.observables, bound_params)
            result = estimator.run([pub]).result()
            cost = cast(_EvaluationData, result[0].data).evs
            vqe_result = VQEIteration(
                iter=out.iter, cost=cost, params=params, result=result
            )
            if out.best is None or cost < out.best.cost:
                out.best = vqe_result
            if callback is not None:
                callback(vqe_result)
            return cost

        try:
            minimize(
                cost_fun,
                self.initial_params,
                method=self.optimizer.scipy_method,
            )
        except StopIteration:
            pass

        return out

    def send(
        self, api: ApiClient, estimator: EstimatorOptions = EstimatorOptions()
    ) -> MaybeAwaitable[JobInfo]:
        return api.create_job(
            q3as.api.JobRequest(
                input=q3as.encoding.EncodedVQE.encode(self), estimator=estimator
            )
        )


class VQEBuilder:
    _ansatz: Optional[QuantumCircuit] = None
    _observables: Optional[ObservablesArrayLike] = None
    _initial_params: Optional[ArrayLike] = None
    _optimizer: Optimizer = COBYLA()
    _maxiter: Optional[int] = None

    def ansatz(self, qc: QuantumCircuit) -> VQEBuilder:
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
        if self._ansatz is None:
            raise ValueError("Ansatz is required")
        if self._observables is None:
            raise ValueError("Observables are required")
        if self._initial_params is not None:
            _initial_params = np.array(self._initial_params)
            if len(_initial_params) != self._ansatz.num_parameters:
                raise ValueError("Initial parameters must match ansatz")
        else:
            _initial_params = np.zeros(self._ansatz.num_parameters)
        return VQE(
            ansatz=self._ansatz,
            observables=ObservablesArray.coerce(self._observables),
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
