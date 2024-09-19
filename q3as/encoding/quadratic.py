from __future__ import annotations
from typing import TYPE_CHECKING, cast, Union, Dict, Tuple, List

from pydantic import BaseModel

import q3as.quadratic.problems as qprob
from q3as.quadratic.problems.types import VarType, QuadraticProgramStatus, ObjSense

if TYPE_CHECKING:
    from q3as.quadratic.problems import (
        QuadraticProgram,
        Variable,
        LinearConstraint,
        QuadraticConstraint,
        QuadraticObjective,
        LinearExpression,
        QuadraticExpression,
    )


class EncodedVariable(BaseModel):
    name: str
    lowerbound: Union[float, int]
    upperbound: Union[float, int]
    vartype: VarType

    @classmethod
    def encode(cls, var: Variable) -> EncodedVariable:
        return cls(
            name=var.name,
            lowerbound=var.lowerbound,
            upperbound=var.upperbound,
            vartype=var.vartype,
        )

    def decode(self, qp: QuadraticProgram) -> Variable:
        return qprob.Variable(
            qp, self.name, self.lowerbound, self.upperbound, self.vartype
        )


type EncodedLinearExpression = Dict[int, float]


def encode_linear_expression(exp: LinearExpression) -> EncodedLinearExpression:
    return cast(EncodedLinearExpression, exp.to_dict())


def decode_linear_expression(
    exp: EncodedLinearExpression,
) -> Dict[Union[int, str], float]:
    return cast(Dict[Union[int, str], float], exp)


class EncodedLinearConstraint(BaseModel):
    name: str
    linear: EncodedLinearExpression
    sense: str
    rhs: float

    @classmethod
    def encode(cls, constraint: LinearConstraint) -> EncodedLinearConstraint:
        return cls(
            name=constraint.name,
            linear=encode_linear_expression(constraint.linear),
            sense=constraint.sense.name,
            rhs=constraint.rhs,
        )

    def decode(self, qp: QuadraticProgram) -> LinearConstraint:
        return qprob.LinearConstraint(
            qp,
            self.name,
            decode_linear_expression(self.linear),
            qprob.constraint.ConstraintSense.convert(self.sense),
            self.rhs,
        )


type EncodedQuadraticExpression = List[Tuple[Tuple[int, int], float]]


def encode_quadratic_expression(exp: QuadraticExpression) -> EncodedQuadraticExpression:
    return cast(EncodedQuadraticExpression, list(exp.to_dict().items()))


def decode_quadratic_expression(
    exp: EncodedQuadraticExpression,
) -> Dict[Tuple[Union[int, str], Union[int, str]], float]:
    return cast(
        Dict[Tuple[Union[int, str], Union[int, str]], float],
        {key: value for key, value in exp},
    )


class EncodedQuadraticConstraint(BaseModel):
    name: str
    linear: EncodedLinearExpression
    quadratic: EncodedQuadraticExpression
    sense: str
    rhs: float

    @classmethod
    def encode(cls, constraint: QuadraticConstraint) -> EncodedQuadraticConstraint:
        return cls(
            name=constraint.name,
            linear=encode_linear_expression(constraint.linear),
            quadratic=encode_quadratic_expression(constraint.quadratic),
            sense=constraint.sense.name,
            rhs=constraint.rhs,
        )

    def decode(self, qp: QuadraticProgram) -> QuadraticConstraint:
        return qprob.QuadraticConstraint(
            qp,
            self.name,
            decode_linear_expression(self.linear),
            decode_quadratic_expression(self.quadratic),
            qprob.constraint.ConstraintSense.convert(self.sense),
            self.rhs,
        )


class EncodedQuadraticObjective(BaseModel):
    constant: float
    linear: EncodedLinearExpression
    quadratic: EncodedQuadraticExpression
    sense: ObjSense

    @classmethod
    def encode(cls, qo: QuadraticObjective) -> EncodedQuadraticObjective:
        return cls(
            constant=qo.constant,
            linear=encode_linear_expression(qo.linear),
            quadratic=encode_quadratic_expression(qo.quadratic),
            sense=qo.sense,
        )

    def decode(self, qp: QuadraticProgram) -> QuadraticObjective:
        return qprob.QuadraticObjective(
            qp,
            self.constant,
            decode_linear_expression(self.linear),
            decode_quadratic_expression(self.quadratic),
            self.sense,
        )


class EncodedQuadraticProgram(BaseModel):
    name: str
    status: QuadraticProgramStatus
    variables: List[EncodedVariable]
    variables_index: Dict[str, int]
    linear_constraints: List[EncodedLinearConstraint]
    linear_constraints_index: Dict[str, int]
    quadratic_constraints: List[EncodedQuadraticConstraint]
    quadratic_constraints_index: Dict[str, int]
    objective: EncodedQuadraticObjective

    @classmethod
    def encode(cls, qp: QuadraticProgram) -> EncodedQuadraticProgram:
        return cls(
            name=qp.name,
            status=qp.status,
            variables=[EncodedVariable.encode(v) for v in qp.variables],
            variables_index=qp.variables_index,
            linear_constraints=[
                EncodedLinearConstraint.encode(v) for v in qp.linear_constraints
            ],
            linear_constraints_index=qp.linear_constraints_index,
            quadratic_constraints=[
                EncodedQuadraticConstraint.encode(v) for v in qp.quadratic_constraints
            ],
            quadratic_constraints_index=qp.quadratic_constraints_index,
            objective=EncodedQuadraticObjective.encode(qp.objective),
        )

    def decode(self) -> QuadraticProgram:
        qp = qprob.QuadraticProgram(self.name)
        qp._status = self.status
        qp._variables = [v.decode(qp) for v in self.variables]
        qp._variables_index = self.variables_index
        qp._linear_constraints = [v.decode(qp) for v in self.linear_constraints]
        qp._linear_constraints_index = self.linear_constraints_index
        qp._quadratic_constraints = [v.decode(qp) for v in self.quadratic_constraints]
        qp._quadratic_constraints_index = self.quadratic_constraints_index
        qp._objective = self.objective.decode(qp)
        return qp
