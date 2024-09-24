import os

from q3as.algo.vqe import VQE, EfficientSU2
from q3as.app import Maxcut
from q3as import Client, Credentials

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator, SamplerV2 as Sampler

from qiskit import transpile

url = os.getenv("Q3AS_URL", "http://localhost:8080")

simulator = AerSimulator()


def simple_maxcut():
    return Maxcut(
        [
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 4, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
        ]
    )


def test_maxcut():
    _ = (
        VQE.builder()
        .app(simple_maxcut())
        .ansatz(EfficientSU2(lambda c: transpile(c, simulator)))
        .run(Estimator(), Sampler())
    )


def test_client_quadratic():
    with Client(Credentials.load(".credentials.json"), url=url) as client:
        _ = VQE.builder().app(simple_maxcut()).send(client)
