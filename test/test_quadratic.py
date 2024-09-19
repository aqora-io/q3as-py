from q3as.algo.vqe import VQE, EfficientSU2
from q3as.quadratic import QuadraticProgram
from q3as.app import Qubo
from q3as.encoding import EncodedVQEResult


from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator, SamplerV2 as Sampler

from qiskit import transpile


simulator = AerSimulator()


def test_quadratic():
    qp = QuadraticProgram("qp")

    qp.integer_var(name="x", lowerbound=-1, upperbound=1)
    qp.integer_var(name="y", lowerbound=-1, upperbound=5)
    qp.integer_var(name="z", lowerbound=-1, upperbound=5)
    qp.minimize(constant=3, linear={"x": 1}, quadratic={("x", "y"): 2, ("z", "z"): -1})

    res = (
        VQE.builder()
        .app(Qubo(qp))
        .ansatz(EfficientSU2(lambda c: transpile(c, simulator)))
        .run(Estimator(), Sampler())
    )
    print(EncodedVQEResult.encode(res).model_dump_json(indent=2))
