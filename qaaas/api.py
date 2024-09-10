from __future__ import annotations
from typing import cast, Literal, Optional, Union
from enum import Enum
import io
import json
import base64
from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel
from qiskit.circuit import QuantumCircuit
from qiskit.primitives.containers.observables_array import (
    ObservablesArray,
    ObservablesArrayLike,
)
from qiskit.primitives.containers.bindings_array import (
    BindingsArray,
    BindingsArrayLike,
)
import qiskit.qpy


class QuantumCircuitEncoding(Enum):
    QPY = "qpy"


class EncodedQuantumCircuit(BaseModel):
    encoding: QuantumCircuitEncoding
    base64: bytes

    @classmethod
    def from_qiskit(cls, qc: QuantumCircuit) -> EncodedQuantumCircuit:
        data = io.BytesIO()
        qiskit.qpy.dump(qc, data)
        return cls(
            encoding=QuantumCircuitEncoding.QPY,
            base64=base64.b64encode(data.getvalue()),
        )

    def to_qiskit(self) -> QuantumCircuit:
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
    def from_qiskit(cls, obs: ObservablesArrayLike) -> EncodedObservables:
        return EncodedObservables(
            data=json.dumps(
                ObservablesArray.coerce(obs).tolist(), separators=(",", ":")
            )
        )

    def to_qiskit(self) -> ObservablesArray:
        return ObservablesArray(json.loads(self.data))


class EncodedArray(BaseModel):
    encoding: Literal["npy"] = "npy"
    base64: bytes

    @classmethod
    def from_numpy(cls, array: ArrayLike) -> EncodedArray:
        data = io.BytesIO()
        np.save(data, array, allow_pickle=False)
        return cls(encoding="npy", base64=base64.b64encode(data.getvalue()))

    def to_numpy(self) -> ArrayLike:
        data = base64.b64decode(self.base64)
        return np.load(io.BytesIO(data), allow_pickle=False)


class Binding(Enum):
    Unbound = "unbound"
    Bound = "bound"


class EncodedParameters(BaseModel):
    binding: Binding
    data: Union[EncodedArray, dict[tuple[str, ...], EncodedArray]]

    @classmethod
    def from_qiskit(cls, params: Union[BindingsArrayLike, ArrayLike]):
        if not isinstance(params, (BindingsArray, Mapping)):
            return cls(binding=Binding.Unbound, data=EncodedArray.from_numpy(params))
        coerced = BindingsArray.coerce(params)
        data = {k: EncodedArray.from_numpy(v) for k, v in coerced.data.items()}
        return cls(binding=Binding.Bound, data=data)

    def to_qiskit(self) -> Union[BindingsArray, ArrayLike]:
        if self.binding == Binding.Unbound:
            if not isinstance(self.data, EncodedArray):
                raise ValueError("Expected unbound data")
            return self.data.to_numpy()
        if not isinstance(self.data, dict):
            raise ValueError("Expected bound data")
        data = {k: v.to_numpy() for k, v in self.data.items()}
        return BindingsArray.coerce(cast(BindingsArrayLike, data))


class COBYLA(BaseModel):
    method: Literal["COBYLA"] = "COBYLA"
    maxiter: Optional[int] = None


class SLSQP(BaseModel):
    method: Literal["SLSQP"] = "SLSQP"
    maxiter: Optional[int] = None


class EstimatorOptions(BaseModel):
    shots: int = 1024


class VQE(BaseModel):
    method: Literal["VQE"] = "VQE"
    ansatz: EncodedQuantumCircuit
    observables: EncodedObservables
    initial_params: Optional[EncodedArray] = None
    optimizer: Union[COBYLA, SLSQP] = COBYLA()
    estimator: EstimatorOptions = EstimatorOptions()

    @classmethod
    def builder(cls) -> VQEBuilder:
        return VQEBuilder()


class VQEBuilder:
    _ansatz: Optional[QuantumCircuit] = None
    _observables: Optional[ObservablesArrayLike] = None
    _initial_params: Optional[ArrayLike] = None
    _optimizer: Union[COBYLA, SLSQP] = COBYLA()
    _estimator: EstimatorOptions = EstimatorOptions()

    def ansatz(self, qc: QuantumCircuit) -> VQEBuilder:
        self._ansatz = qc
        return self

    def observables(self, obs: ObservablesArrayLike) -> VQEBuilder:
        self._observables = obs
        return self

    def initial_params(self, params: ArrayLike) -> "VQEBuilder":
        self._initial_params = params
        return self

    def optimizer(self, opt: Union[COBYLA, SLSQP]) -> VQEBuilder:
        self._optimizer = opt
        return self

    def estimator(self, est: EstimatorOptions) -> VQEBuilder:
        self._estimator = est
        return self

    def build(self) -> VQE:
        if self._ansatz is None:
            raise ValueError("Ansatz is required")
        if self._observables is None:
            raise ValueError("Observables are required")
        return VQE(
            ansatz=EncodedQuantumCircuit.from_qiskit(self._ansatz),
            observables=EncodedObservables.from_qiskit(self._observables),
            initial_params=(
                EncodedArray.from_numpy(self._initial_params)
                if self._initial_params is not None
                else None
            ),
            optimizer=self._optimizer,
            estimator=self._estimator,
        )
