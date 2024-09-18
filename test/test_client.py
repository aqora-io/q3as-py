import os

import numpy as np
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp

from q3as import Client, Credentials, VQE

url = os.getenv("Q3AS_URL", "http://localhost:8080")


async def test_client_with_init_params():
    hamiltonian = SparsePauliOp.from_list(
        [("YZ", 0.3980), ("ZI", -0.3980), ("ZZ", -0.0113), ("XX", 0.1810)]
    )
    ansatz = ansatz = EfficientSU2(hamiltonian.num_qubits)
    x0 = 2 * np.pi * np.random.random(ansatz.num_parameters)
    with Client(Credentials.load(".credentials.json"), url=url) as client:
        response = (
            VQE.builder()
            .ansatz(ansatz)
            .observables(hamiltonian)
            .initial_params(x0)
            .send(client)
        )
        print(response)


async def test_client_no_init_params():
    hamiltonian = SparsePauliOp.from_list(
        [("YZ", 0.3980), ("ZI", -0.3980), ("ZZ", -0.0113), ("XX", 0.1810)]
    )
    ansatz = ansatz = EfficientSU2(hamiltonian.num_qubits)
    with Client(Credentials.load(".credentials.json"), url=url) as client:
        response = VQE.builder().ansatz(ansatz).observables(hamiltonian).send(client)
        print(response)
