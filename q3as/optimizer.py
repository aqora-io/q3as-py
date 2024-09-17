from abc import ABC, abstractmethod


class Optimizer(ABC):
    @property
    @abstractmethod
    def scipy_method(self) -> str:
        pass


class COBYLA(Optimizer):
    @property
    def scipy_method(self) -> str:
        return "COBYLA"


class SLSQP(Optimizer):
    @property
    def scipy_method(self) -> str:
        return "SLSQP"
