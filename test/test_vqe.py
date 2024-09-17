from q3as.algo.vqe import VQE

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit import transpile
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp

simulator = AerSimulator()


def test_vqe_with_init_params():
    hamiltonian = SparsePauliOp.from_list(
        [("YZ", 0.3980), ("ZI", -0.3980), ("ZZ", -0.0113), ("XX", 0.1810)]
    )
    ansatz = ansatz = EfficientSU2(hamiltonian.num_qubits)
    x0 = 2 * np.pi * np.random.random(ansatz.num_parameters)
    vqe = (
        VQE.builder()
        .ansatz(transpile(ansatz, simulator))
        .observables(hamiltonian)
        .initial_params(x0)
        .build()
    )
    vqe.run(Estimator())


def test_vqe_no_init_params():
    hamiltonian = SparsePauliOp.from_list(
        [("YZ", 0.3980), ("ZI", -0.3980), ("ZZ", -0.0113), ("XX", 0.1810)]
    )
    ansatz = ansatz = EfficientSU2(hamiltonian.num_qubits)
    vqe = (
        VQE.builder()
        .ansatz(transpile(ansatz, simulator))
        .observables(hamiltonian)
        .build()
    )
    vqe.run(Estimator())


def test_vqe_with_maxiter():
    hamiltonian = SparsePauliOp.from_list(
        [("YZ", 0.3980), ("ZI", -0.3980), ("ZZ", -0.0113), ("XX", 0.1810)]
    )
    ansatz = ansatz = EfficientSU2(hamiltonian.num_qubits)
    vqe = (
        VQE.builder()
        .ansatz(transpile(ansatz, simulator))
        .observables(hamiltonian)
        .maxiter(100)
        .build()
    )
    vqe.run(Estimator())
