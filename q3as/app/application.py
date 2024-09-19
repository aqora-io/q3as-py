from __future__ import annotations
from typing import TypeVar, Generic, Literal, List, Tuple
from abc import ABC, abstractmethod

import numpy as np

from qiskit.primitives import BitArray
from qiskit.primitives.containers.observables_array import (
    ObservablesArrayLike,
)

type ApplicationName = Literal["qubo"]


class BitString:
    data: str

    def __init__(self, data: str) -> None:
        self.data = data

    def to_list(self) -> np.ndarray:
        return np.array([b == "1" for b in self.data], dtype=bool)


Encoded = TypeVar("Encoded")
State = TypeVar("State")


class Application(ABC, Generic[Encoded, State]):
    @abstractmethod
    def name(self) -> ApplicationName:
        pass

    @abstractmethod
    def encode(self) -> Encoded:
        pass

    @classmethod
    @abstractmethod
    def decode(cls, encoded: Encoded) -> Application:
        pass

    @abstractmethod
    def hamiltonian(self) -> ObservablesArrayLike:
        pass

    @abstractmethod
    def interpret(self, bit_string: BitString) -> State:
        pass

    def interpreted_counts(self, meas: BitArray) -> List[Tuple[State, int]]:
        out = {}
        for state, count in meas.get_counts().items():
            interpreted = self.interpret(BitString(state))
            cur_count = out.get(interpreted, 0)
            out[interpreted] = cur_count + count
        return list(out.items())
