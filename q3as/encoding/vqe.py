from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any, List, Tuple

from pydantic import BaseModel

import q3as.algo.vqe

from q3as.encoding.numpy import EncodedArray
from q3as.encoding.qiskit import (
    EncodedEstimatorResult,
    EncodedSamplerResult,
    EncodedObservables,
    EncodedQuantumCircuit,
)

from .app import EncodedApplication
from .optimizer import EncodedOptimizer
from q3as.algo.types import HaltReason

if TYPE_CHECKING:
    from q3as.algo.vqe import VQE, VQEResult, VQEIteration


class EncodedVQEIteration(BaseModel):
    iter: int
    cost: float
    params: EncodedArray
    estimated: EncodedEstimatorResult
    best: bool

    @classmethod
    def encode(cls, result: VQEIteration) -> EncodedVQEIteration:
        return cls(
            iter=result.iter,
            cost=result.cost,
            params=EncodedArray.encode(result.params),
            estimated=EncodedEstimatorResult.encode(result.estimated),
            best=result.best,
        )

    def decode(self) -> VQEIteration:
        return q3as.algo.vqe.VQEIteration(
            iter=self.iter,
            cost=self.cost,
            params=self.params.decode(),
            estimated=self.estimated.decode(),
            best=self.best,
        )


class EncodedVQEResult(BaseModel):
    params: EncodedArray
    iter: int
    reason: HaltReason
    cost: Optional[float]
    estimated: Optional[EncodedEstimatorResult]
    sampled: Optional[EncodedSamplerResult]
    interpreted: Optional[List[Tuple[Any, int]]]

    @classmethod
    def encode(cls, result: VQEResult) -> EncodedVQEResult:
        return cls(
            params=EncodedArray.encode(result.params),
            iter=result.iter,
            reason=result.reason,
            cost=result.cost,
            estimated=None
            if result.estimated is None
            else EncodedEstimatorResult.encode(result.estimated),
            sampled=None
            if result.sampled is None
            else EncodedSamplerResult.encode(result.sampled),
            interpreted=result.interpreted,
        )

    def decode(self):
        return q3as.algo.vqe.VQEResult(
            iter=self.iter,
            cost=self.cost,
            params=self.params.decode(),
            reason=self.reason,
            estimated=None if self.estimated is None else self.estimated.decode(),
            sampled=None if self.sampled is None else self.sampled.decode(),
            interpreted=self.interpreted,
        )


class EncodedVQE(BaseModel):
    ansatz: EncodedQuantumCircuit
    observables: EncodedObservables
    initial_params: EncodedArray
    optimizer: EncodedOptimizer
    maxiter: int
    app: Optional[EncodedApplication]

    @classmethod
    def encode(cls, vqe: VQE) -> EncodedVQE:
        return cls(
            ansatz=EncodedQuantumCircuit.encode(vqe.ansatz),
            observables=EncodedObservables.encode(vqe.observables),
            initial_params=EncodedArray.encode(vqe.initial_params),
            optimizer=EncodedOptimizer.encode(vqe.optimizer),
            maxiter=vqe.maxiter,
            app=None if vqe.app is None else EncodedApplication.encode(vqe.app),
        )

    def decode(self) -> VQE:
        return q3as.algo.vqe.VQE(
            ansatz=self.ansatz.decode(),
            observables=self.observables.decode(),
            initial_params=self.initial_params.decode(),
            optimizer=self.optimizer,
            maxiter=self.maxiter,
            app=None if self.app is None else self.app.decode(),
        )
