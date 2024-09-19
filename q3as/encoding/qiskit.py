from __future__ import annotations
from typing import cast, Literal, Dict, Any, List
from enum import Enum
import io
import json
import base64

from pydantic import BaseModel
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import PrimitiveResult, DataBin, BitArray
from qiskit.primitives.containers.observables_array import (
    ObservablesArray,
    ObservablesArrayLike,
)
from qiskit.primitives.containers.pub_result import PubResult
from qiskit.primitives.containers.sampler_pub_result import SamplerPubResult
import qiskit.qpy

from .numpy import EncodedArray

from q3as.algo.types import EstimatorData, SamplerData


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


class EncodedBitArray(BaseModel):
    array: EncodedArray
    num_bits: int

    @classmethod
    def encode(cls, array: BitArray) -> EncodedBitArray:
        return cls(array=EncodedArray.encode(array.array), num_bits=array.num_bits)

    def decode(self) -> BitArray:
        return BitArray(array=self.array.decode(), num_bits=self.num_bits)


class EncodedEstimatorPubResult(BaseModel):
    metadata: Dict[str, Any]
    evs: float
    stds: float

    @classmethod
    def encode(cls, result: PubResult) -> EncodedEstimatorPubResult:
        data = cast(EstimatorData, result.data)
        return cls(metadata=result.metadata, evs=data.evs, stds=data.stds)

    def decode(self) -> PubResult:
        return PubResult(
            metadata=self.metadata, data=DataBin(evs=self.evs, stds=self.stds)
        )


class EncodedSamplerPubResult(BaseModel):
    metadata: Dict[str, Any]
    meas: EncodedBitArray

    @classmethod
    def encode(cls, result: SamplerPubResult) -> EncodedSamplerPubResult:
        data = cast(SamplerData, result.data)
        return cls(metadata=result.metadata, meas=EncodedBitArray.encode(data.meas))

    def decode(self) -> SamplerPubResult:
        return SamplerPubResult(
            metadata=self.metadata, data=DataBin(meas=self.meas.decode())
        )


class EncodedEstimatorResult(BaseModel):
    metadata: Dict[str, Any]
    pub_results: List[EncodedEstimatorPubResult]

    @classmethod
    def encode(cls, result: PrimitiveResult[PubResult]) -> EncodedEstimatorResult:
        return cls(
            metadata=result.metadata,
            pub_results=[EncodedEstimatorPubResult.encode(r) for r in iter(result)],
        )

    def decode(self) -> PrimitiveResult[PubResult]:
        return PrimitiveResult(
            pub_results=[r.decode() for r in self.pub_results],
            metadata=self.metadata,
        )


class EncodedSamplerResult(BaseModel):
    metadata: Dict[str, Any]
    pub_results: List[EncodedSamplerPubResult]

    @classmethod
    def encode(cls, result: PrimitiveResult[SamplerPubResult]) -> EncodedSamplerResult:
        return cls(
            metadata=result.metadata,
            pub_results=[EncodedSamplerPubResult.encode(r) for r in iter(result)],
        )

    def decode(self) -> PrimitiveResult[SamplerPubResult]:
        return PrimitiveResult(
            pub_results=[r.decode() for r in self.pub_results],
            metadata=self.metadata,
        )
