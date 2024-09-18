from enum import Enum


class QuadraticProgramStatus(Enum):
    """Status of QuadraticProgram"""

    VALID = 0
    INFEASIBLE = 1


class VarType(Enum):
    """Constants defining variable type."""

    CONTINUOUS = 0
    BINARY = 1
    INTEGER = 2


class ObjSense(Enum):
    """Objective Sense Type."""

    MINIMIZE = 1
    MAXIMIZE = -1
