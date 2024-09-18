from __future__ import annotations
from typing import Literal, Dict, Any, List
from enum import Enum
import io
import json
import base64

from pydantic import BaseModel
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import PrimitiveResult, DataBin
from qiskit.primitives.containers.observables_array import (
    ObservablesArray,
    ObservablesArrayLike,
)
from qiskit.primitives.containers.pub_result import PubResult
import qiskit.qpy

from .numpy import EncodedArray


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


class EncodedEstimatorResult(BaseModel):
    metadata: Dict[str, Any]
    pub_results: List[EncodedPubResult]

    @classmethod
    def encode(cls, result: PrimitiveResult[PubResult]) -> EncodedEstimatorResult:
        return cls(
            metadata=result.metadata,
            pub_results=[EncodedPubResult.encode(r) for r in iter(result)],
        )

    def decode(self) -> PrimitiveResult[PubResult]:
        return PrimitiveResult(
            pub_results=[r.decode() for r in self.pub_results],
            metadata=self.metadata,
        )
