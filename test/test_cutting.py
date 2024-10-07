from typing import cast
import numpy as np

from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp

# from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import EfficientSU2
from qiskit.result import QuasiDistribution

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2, SamplerV2, Sampler

from q3as.algo.cutting import (
    find_cut_actions,
    create_estimate_cut,
    create_sample_cut,
)
from q3as.algo.types import EstimatorData


def test_cutting_estimate():
    observable = SparsePauliOp(["ZIIIIII", "IIIZIII", "IIIIIIZ"])
    # circuit = random_circuit(7, 6, max_operands=2, seed=1242)
    circuit = EfficientSU2(observable.num_qubits, reps=2)
    params = np.zeros(circuit.num_parameters)
    params[0] = np.pi / 2
    params[2] = np.pi / 2

    circuit = transpile(circuit, AerSimulator())
    found = find_cut_actions(circuit, 4, seed=1)
    experiments = create_estimate_cut(circuit, observable, found.actions)
    sampler = SamplerV2()
    cut_val, samples = experiments.run(sampler, params)
    assert len(samples) == 2

    uncut_val = cast(
        EstimatorData,
        EstimatorV2()
        .run([(circuit, observable, {tuple(circuit.parameters): params})])
        .result()[0]
        .data,
    ).evs
    assert abs(cut_val - uncut_val) <= 0.25


def test_cutting_sample():
    observable = SparsePauliOp(["ZIIIIII", "IIIZIII", "IIIIIIZ"])
    # circuit = random_circuit(7, 6, max_operands=2, seed=1242)
    circuit = EfficientSU2(observable.num_qubits, reps=2)

    sampler = SamplerV2()

    params = np.zeros(circuit.num_parameters)
    params[0] = np.pi / 2
    params[2] = np.pi / 2

    circuit = transpile(circuit, AerSimulator())

    measured_circuit = circuit.measure_all(inplace=False)
    assert measured_circuit is not None
    result = sampler.run(
        [(measured_circuit, {tuple(measured_circuit.parameters): params})]
    ).result()[0]
    counts = result.data.meas.get_counts()
    num_shots = result.data.meas.num_shots
    quasi_dist = QuasiDistribution(
        {key: count / num_shots for key, count in counts.items()}
    )
    print(quasi_dist)

    # print()
    # print(circuit.draw())
    found = find_cut_actions(circuit, 4, seed=1)
    experiments = create_sample_cut(circuit, found.actions)
    # for experiment in experiments.subexperiments.values():
    #     for subexperiment in experiment:
    #         print(subexperiment.draw())
    cool = experiments.run(sampler, params)
    print(cool)
