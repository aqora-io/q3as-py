from q3as.algo.vqe import VQE
from q3as.quadratic import QuadraticProgram
from q3as.app import Qubo
from q3as.encoding import EncodedVQE


def simple_qp():
    qp = QuadraticProgram("qp")

    qp.integer_var(name="x", lowerbound=-1, upperbound=1)
    qp.integer_var(name="y", lowerbound=-1, upperbound=5)
    qp.integer_var(name="z", lowerbound=-1, upperbound=5)
    qp.minimize(constant=3, linear={"x": 1}, quadratic={("x", "y"): 2, ("z", "z"): -1})

    return qp


def test_encoding_quadratic():
    vqe = VQE.builder().app(Qubo(simple_qp())).build()
    encoded = EncodedVQE.encode(vqe).model_dump_json()
    print(encoded)
    decoded = EncodedVQE.decode(EncodedVQE.model_validate_json(encoded))
    print(decoded.app.program.prettyprint())
