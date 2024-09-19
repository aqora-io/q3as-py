from q3as.algo.vqe import VQE, HaltReason, VQEIteration, QAOA

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator, SamplerV2 as Sampler
from qiskit import transpile
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp

simulator = AerSimulator()


def test_vqe_with_init_params():
    hamiltonian = SparsePauliOp.from_list(
        [("YZ", 0.3980), ("ZI", -0.3980), ("ZZ", -0.0113), ("XX", 0.1810)]
    )
    ansatz = EfficientSU2(hamiltonian.num_qubits)
    x0 = 2 * np.pi * np.random.random(ansatz.num_parameters)
    vqe = (
        VQE.builder()
        .ansatz(transpile(ansatz, simulator))
        .observables(hamiltonian)
        .initial_params(x0)
        .build()
    )
    res = vqe.run(Estimator())
    assert res.reason == HaltReason.TOLERANCE


def test_vqe_no_init_params():
    hamiltonian = SparsePauliOp.from_list(
        [("YZ", 0.3980), ("ZI", -0.3980), ("ZZ", -0.0113), ("XX", 0.1810)]
    )
    ansatz = EfficientSU2(hamiltonian.num_qubits)
    vqe = (
        VQE.builder()
        .ansatz(transpile(ansatz, simulator))
        .observables(hamiltonian)
        .build()
    )
    res = vqe.run(Estimator())
    assert res.reason == HaltReason.TOLERANCE


def test_vqe_with_maxiter():
    hamiltonian = SparsePauliOp.from_list(
        [("YZ", 0.3980), ("ZI", -0.3980), ("ZZ", -0.0113), ("XX", 0.1810)]
    )
    ansatz = EfficientSU2(hamiltonian.num_qubits)
    vqe = (
        VQE.builder()
        .ansatz(transpile(ansatz, simulator))
        .observables(hamiltonian)
        .maxiter(100)
        .build()
    )
    res = vqe.run(Estimator())
    assert res.reason == HaltReason.MAXITER


def test_vqe_with_interrupt():
    hamiltonian = SparsePauliOp.from_list(
        [("YZ", 0.3980), ("ZI", -0.3980), ("ZZ", -0.0113), ("XX", 0.1810)]
    )
    ansatz = EfficientSU2(hamiltonian.num_qubits)
    vqe = (
        VQE.builder()
        .ansatz(transpile(ansatz, simulator))
        .observables(hamiltonian)
        .maxiter(100)
        .build()
    )

    def callback(iter: VQEIteration):
        if iter.iter >= 10:
            raise StopIteration

    res = vqe.run(Estimator(), callback=callback)
    assert res.reason == HaltReason.INTERRUPT


def test_vqe_with_callable_ansatz():
    hamiltonian = SparsePauliOp.from_list(
        [("YZ", 0.3980), ("ZI", -0.3980), ("ZZ", -0.0113), ("XX", 0.1810)]
    )
    res = (
        VQE.builder()
        .ansatz(QAOA(lambda qc: transpile(qc, simulator)))
        .observables(hamiltonian)
        .run(Estimator())
    )
    assert res.reason == HaltReason.TOLERANCE


def test_vqe_with_sampler():
    hamiltonian = SparsePauliOp.from_list(
        [("YZ", 0.3980), ("ZI", -0.3980), ("ZZ", -0.0113), ("XX", 0.1810)]
    )
    res = (
        VQE.builder()
        .ansatz(QAOA(lambda qc: transpile(qc, simulator)))
        .observables(hamiltonian)
        .run(Estimator(), Sampler())
    )
    assert res.reason == HaltReason.TOLERANCE
