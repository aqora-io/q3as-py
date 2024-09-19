from enum import Enum

from qiskit.primitives import BitArray


class HaltReason(Enum):
    TOLERANCE = "tolerance"
    INTERRUPT = "interrupt"
    MAXITER = "maxiter"


class EstimatorData:
    evs: float
    stds: float


class SamplerData:
    meas: BitArray
