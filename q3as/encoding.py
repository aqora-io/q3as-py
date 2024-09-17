from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Dict, Any, List, Optional
from enum import Enum
import io
import json
import base64

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import PrimitiveResult, DataBin
from qiskit.primitives.containers.observables_array import (
    ObservablesArray,
    ObservablesArrayLike,
)
from qiskit.primitives.containers.pub_result import PubResult
import qiskit.qpy

import q3as.algo.vqe
from q3as.optimizer import Optimizer

if TYPE_CHECKING:
    from q3as.algo.vqe import VQE, VQEResult, VQEIteration


class QuantumCircuitEncoding(Enum):
    QPY = "qpy"


class EncodedQuantumCircuit(BaseModel):
    encoding: QuantumCircuitEncoding
    base64: bytes

    @classmethod
    def encode(cls, qc: QuantumCircuit) -> EncodedQuantumCircuit:
        data = io.BytesIO()
        qiskit.qpy.dump(qc, data)
        return cls(
            encoding=QuantumCircuitEncoding.QPY,
            base64=base64.b64encode(data.getvalue()),
        )

    def decode(self) -> QuantumCircuit:
        if self.encoding != QuantumCircuitEncoding.QPY:
            raise ValueError("Only QPY encoding is supported")
        data = base64.b64decode(self.base64)
        loaded = qiskit.qpy.load(io.BytesIO(data))
        if len(loaded) != 1:
            raise ValueError("Expected single quantum circuit")
        if not isinstance(loaded[0], QuantumCircuit):
            raise ValueError("Expected quantum circuit")
        return loaded[0]


class EncodedObservables(BaseModel):
    encoding: Literal["JSON"] = "JSON"
    data: str

    @classmethod
    def encode(cls, obs: ObservablesArrayLike) -> EncodedObservables:
        return EncodedObservables(
            data=json.dumps(
                ObservablesArray.coerce(obs).tolist(), separators=(",", ":")
            )
        )

    def decode(self) -> ObservablesArray:
        return ObservablesArray(json.loads(self.data))


class EncodedArray(BaseModel):
    encoding: Literal["npy"] = "npy"
    base64: bytes

    @classmethod
    def encode(cls, array: ArrayLike) -> EncodedArray:
        data = io.BytesIO()
        np.save(data, array, allow_pickle=False)
        return cls(encoding="npy", base64=base64.b64encode(data.getvalue()))

    def decode(self) -> np.ndarray:
        data = base64.b64decode(self.base64)
        return np.load(io.BytesIO(data), allow_pickle=False)


class EncodedPubResult(BaseModel):
    metadata: Dict[str, Any]
    data: Dict[str, EncodedArray]

    @classmethod
    def encode(cls, result: PubResult) -> EncodedPubResult:
        return cls(
            metadata=result.metadata,
            data={k: EncodedArray.encode(v) for k, v in result.data.items()},
        )

    def decode(self) -> PubResult:
        return PubResult(
            metadata=self.metadata,
            data=DataBin(**{k: v.decode() for k, v in self.data.items()}),
        )


class EncodedResult(BaseModel):
    metadata: Dict[str, Any]
    pub_results: List[EncodedPubResult]

    @classmethod
    def encode(cls, result: PrimitiveResult[PubResult]) -> EncodedResult:
        return cls(
            metadata=result.metadata,
            pub_results=[EncodedPubResult.encode(r) for r in iter(result)],
        )

    def decode(self) -> PrimitiveResult[PubResult]:
        return PrimitiveResult(
            pub_results=[r.decode() for r in self.pub_results],
            metadata=self.metadata,
        )


class EncodedVQEIteration(BaseModel):
    iter: int
    cost: float
    params: EncodedArray
    result: EncodedResult

    @classmethod
    def encode(cls, result: VQEIteration) -> EncodedVQEIteration:
        return cls(
            iter=result.iter,
            cost=result.cost,
            params=EncodedArray.encode(result.params),
            result=EncodedResult.encode(result.result),
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


class EncodedOptimizer(BaseModel, Optimizer):
    method: str

    @classmethod
    def encode(cls, optimizer: Optimizer) -> EncodedOptimizer:
        return cls(method=optimizer.scipy_method)

    @property
    def scipy_method(self) -> str:
        return self.method


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
