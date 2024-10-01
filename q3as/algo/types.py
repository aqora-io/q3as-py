from enum import Enum

from qiskit.primitives import BitArray


class HaltReason(Enum):
    """
    The reason for halting
    """

    TOLERANCE = "tolerance"
    "The tolerance was reached"
    INTERRUPT = "interrupt"
    "The process was interrupted"
    MAXITER = "maxiter"
    "The maximum number of iterations was reached"


class EstimatorData:
    evs: float
    stds: float


class SamplerData:
    meas: BitArray
