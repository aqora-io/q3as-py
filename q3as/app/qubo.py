from __future__ import annotations
from typing import Tuple, Any

import numpy as np

from q3as.quadratic import QuadraticProgram
from q3as.quadratic.converters import QuadraticProgramToQubo
from q3as.encoding.quadratic import EncodedQuadraticProgram

from .application import Application, ApplicationName, BitString


class Qubo(Application[EncodedQuadraticProgram, frozenset[Tuple[str, float]]]):
    def __init__(self, qp: QuadraticProgram):
        """
        Create a new QUBO application.
        """
        self.program = qp
        self.converter = QuadraticProgramToQubo()
        self.converted = self.converter.convert(qp)

    def name(self) -> ApplicationName:
        return "qubo"

    def encode(self) -> EncodedQuadraticProgram:
        return EncodedQuadraticProgram.encode(self.program)

    @classmethod
    def validate_encoded(cls, encoded: Any) -> EncodedQuadraticProgram:
        return EncodedQuadraticProgram.model_validate(encoded)

    @classmethod
    def decode(cls, encoded: EncodedQuadraticProgram) -> Qubo:
        return Qubo(EncodedQuadraticProgram.decode(encoded))

    def hamiltonian(self):
        """
        Return the Hamiltonian of the QUBO problem.
        """
        return self.converted.to_ising()[0]

    def interpret(self, bit_string: BitString) -> frozenset[Tuple[str, float]]:
        """
        Interpret the bit string as a solution to the QUBO problem. This will return a list of pairs of variable names and their assigned values
        """
        values = self.converter.interpret(np.flip(bit_string.to_list()))
        return frozenset(
            [
                (name, float(values[index]))
                for name, index in self.program.variables_index.items()
            ]
        )
