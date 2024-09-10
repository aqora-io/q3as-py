from qaaas.client import Client, Credentials
from qaaas.api import VQE

import numpy as np
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp


async def test_client():
    hamiltonian = SparsePauliOp.from_list(
        [("YZ", 0.3980), ("ZI", -0.3980), ("ZZ", -0.0113), ("XX", 0.1810)]
    )
    ansatz = ansatz = EfficientSU2(hamiltonian.num_qubits)
    x0 = 2 * np.pi * np.random.random(ansatz.num_parameters)
    vqe = (
        VQE.builder().ansatz(ansatz).observables(hamiltonian).initial_params(x0).build()
    )
    async with Client(
        Credentials.load(open(".credentials.json")), url="http://localhost:8000"
    ) as client:
        response = await client.create_job(vqe)
        print(response)