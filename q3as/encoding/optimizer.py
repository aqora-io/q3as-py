from __future__ import annotations

from pydantic import BaseModel

from q3as.algo.optimizer import Optimizer


class EncodedOptimizer(BaseModel, Optimizer):
    method: str

    @classmethod
    def encode(cls, optimizer: Optimizer) -> EncodedOptimizer:
        return cls(method=optimizer.scipy_method)

    @property
    def scipy_method(self) -> str:
        return self.method
