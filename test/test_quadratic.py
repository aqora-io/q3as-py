import os

from q3as.algo.vqe import VQE, EfficientSU2
from q3as.quadratic import QuadraticProgram
from q3as.app import Qubo
from q3as import Client, Credentials


from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator, SamplerV2 as Sampler

from qiskit import transpile

url = os.getenv("Q3AS_URL", "http://localhost:8080")

simulator = AerSimulator()


def simple_qp():
    qp = QuadraticProgram("qp")

    qp.integer_var(name="x", lowerbound=-1, upperbound=1)
    qp.integer_var(name="y", lowerbound=-1, upperbound=5)
    qp.integer_var(name="z", lowerbound=-1, upperbound=5)
    qp.minimize(constant=3, linear={"x": 1}, quadratic={("x", "y"): 2, ("z", "z"): -1})

    return qp


def test_quadratic():
    _ = (
        VQE.builder()
        .app(Qubo(simple_qp()))
        .ansatz(EfficientSU2(lambda c: transpile(c, simulator)))
        .run(Estimator(), Sampler())
    )


def test_client_quadratic():
    with Client(Credentials.load(".credentials.json"), url=url) as client:
        _ = VQE.builder().app(Qubo(simple_qp())).send(client)
