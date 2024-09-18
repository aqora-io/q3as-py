from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

import q3as.algo.vqe

from q3as.encoding.numpy import EncodedArray
from q3as.encoding.qiskit import (
    EncodedEstimatorResult,
    EncodedObservables,
    EncodedQuantumCircuit,
)

from .optimizer import EncodedOptimizer

if TYPE_CHECKING:
    from q3as.algo.vqe import VQE, VQEResult, VQEIteration


class EncodedVQEIteration(BaseModel):
    iter: int
    cost: float
    params: EncodedArray
    result: EncodedEstimatorResult

    @classmethod
    def encode(cls, result: VQEIteration) -> EncodedVQEIteration:
        return cls(
            iter=result.iter,
            cost=result.cost,
            params=EncodedArray.encode(result.params),
            result=EncodedEstimatorResult.encode(result.result),
        )

    def decode(self) -> VQEIteration:
        return q3as.algo.vqe.VQEIteration(
            iter=self.iter,
            cost=self.cost,
            params=self.params.decode(),
            result=self.result.decode(),
        )


class EncodedVQEResult(BaseModel):
    iter: int
    best: Optional[EncodedVQEIteration]

    @classmethod
    def encode(cls, result: VQEResult) -> EncodedVQEResult:
        return cls(
            iter=result.iter,
            best=None
            if result.best is None
            else EncodedVQEIteration.encode(result.best),
        )

    def decode(self):
        return q3as.algo.vqe.VQEResult(
            iter=self.iter,
            best=None if self.best is None else self.best.decode(),
        )


class EncodedVQE(BaseModel):
    ansatz: EncodedQuantumCircuit
    observables: EncodedObservables
    initial_params: EncodedArray
    optimizer: EncodedOptimizer
    maxiter: Optional[int]

    @classmethod
    def encode(cls, vqe: VQE) -> EncodedVQE:
        return cls(
            ansatz=EncodedQuantumCircuit.encode(vqe.ansatz),
            observables=EncodedObservables.encode(vqe.observables),
            initial_params=EncodedArray.encode(vqe.initial_params),
            optimizer=EncodedOptimizer.encode(vqe.optimizer),
            maxiter=vqe.maxiter,
        )

    def decode(self) -> VQE:
        return q3as.algo.vqe.VQE(
            ansatz=self.ansatz.decode(),
            observables=self.observables.decode(),
            initial_params=self.initial_params.decode(),
            optimizer=self.optimizer,
            maxiter=self.maxiter,
        )
