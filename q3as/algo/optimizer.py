from abc import ABC, abstractmethod


class Optimizer(ABC):
    """An optimizer for variational algorithms."""

    @property
    @abstractmethod
    def scipy_method(self) -> str:
        """The name of the optimizer method in scipy.optimize."""
        pass


class COBYLA(Optimizer):
    @property
    def scipy_method(self) -> str:
        return "COBYLA"


class SLSQP(Optimizer):
    @property
    def scipy_method(self) -> str:
        return "SLSQP"
