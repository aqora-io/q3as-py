from typing import cast, Literal, Optional, List, Dict, Hashable, Tuple
from pydantic import BaseModel
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qiskit.circuit import QuantumCircuit, CircuitInstruction
from qiskit.quantum_info import SparsePauliOp, PauliList
from qiskit.primitives import BaseSamplerV2, PrimitiveResult, SamplerResult

from qiskit_addon_cutting.automated_cut_finding import (
    DeviceConstraints,
)
from qiskit_addon_cutting.cutting_experiments import WeightType
from qiskit_addon_cutting.instructions import CutWire
from qiskit_addon_cutting import (
    partition_problem,
    cut_wires,
    expand_observables,
    generate_cutting_experiments,
    reconstruct_expectation_values,
    reconstruct_distribution,
)

from qiskit_addon_cutting.cutting_decomposition import cut_gates
from qiskit_addon_cutting.cut_finding.optimization_settings import OptimizationSettings
from qiskit_addon_cutting.cut_finding.disjoint_subcircuits_state import (
    DisjointSubcircuitsState,
)
from qiskit_addon_cutting.cut_finding.circuit_interface import SimpleGateList
from qiskit_addon_cutting.cut_finding.lo_cuts_optimizer import LOCutsOptimizer
from qiskit_addon_cutting.cut_finding.cco_utils import qc_to_cco_circuit


class CutAction(BaseModel):
    name: Literal["CutTwoQubitGate", "CutLeftWire", "CutRightWire", "CutBothWires"]
    instruction_id: int
    qubit1_id: Optional[int] = None
    qubit2_id: Optional[int] = None


class FoundCutActions(BaseModel):
    sampling_overhead: float
    minimum_reached: bool
    actions: List[CutAction]


def find_cut_actions(
    circuit: QuantumCircuit,
    num_qubits: int,
    *,
    seed: Optional[int] = None,
    max_iters: int = 10000,
):
    circuit_cco = qc_to_cco_circuit(circuit)
    interface = SimpleGateList(circuit_cco)

    constraints = DeviceConstraints(num_qubits)
    opt_settings = OptimizationSettings(
        seed=seed,
        max_backjumps=max_iters,
    )

    # Hard-code the optimizer to an LO-only optimizer
    optimizer = LOCutsOptimizer(interface, opt_settings, constraints)

    # Find cut locations
    opt_out = optimizer.optimize()

    actions = []

    opt_out = cast(DisjointSubcircuitsState, opt_out)
    opt_out.actions = opt_out.actions or []
    for action in opt_out.actions:
        if action.action.get_name() == "CutTwoQubitGate":
            actions.append(
                CutAction(
                    name="CutTwoQubitGate",
                    instruction_id=action.gate_spec.instruction_id,
                )
            )
        else:
            name = action.action.get_name()
            assert name in (
                "CutBothWires",
                "CutLeftWire",
                "CutRightWire",
            )
            new_action = CutAction(
                name=name,
                instruction_id=action.gate_spec.instruction_id,
                qubit1_id=action.args[0][0] - 1,
            )
            if name == "CutBothWires":
                assert len(action.args) == 2
                new_action.qubit2_id = action.args[1][0] - 1
            actions.append(new_action)

    return FoundCutActions(
        sampling_overhead=opt_out.upper_bound_gamma() ** 2,
        minimum_reached=optimizer.minimum_reached(),
        actions=actions,
    )


def cut_circuit(circuit: QuantumCircuit, actions: List[CutAction]):
    cut_gate_ids = [
        action.instruction_id for action in actions if action.name == "CutTwoQubitGate"
    ]

    circ_out = cut_gates(circuit, cut_gate_ids)[0]

    wire_cut_actions = [
        action for action in actions if action.name != "CutTwoQubitGate"
    ]

    # Insert all the wire cuts
    counter = 0
    for action in sorted(wire_cut_actions, key=lambda a: a.instruction_id):
        assert action.qubit1_id is not None
        circ_out.data.insert(
            action.instruction_id + counter,
            CircuitInstruction(
                CutWire(),
                [circuit.data[action.instruction_id].qubits[action.qubit1_id]],
                [],
            ),
        )
        counter += 1

        if action.name == "CutBothWires":
            assert action.qubit2_id is not None
            circ_out.data.insert(
                action.instruction_id + counter,
                CircuitInstruction(
                    CutWire(),
                    [circuit.data[action.instruction_id].qubits[action.qubit2_id]],
                    [],
                ),
            )
            counter += 1

    return cut_wires(circ_out)


@dataclass
class EstimateCut:
    subobservables: Dict[Hashable, PauliList]
    subexperiments: Dict[Hashable, List[QuantumCircuit]]
    param_map: Dict[Hashable, int]
    coefficients: NDArray
    exp_coefficients: List[Tuple[float, WeightType]]

    def run(self, sampler: BaseSamplerV2, parameters: NDArray):
        jobs = {
            label: sampler.run(
                [
                    (
                        subexperiment,
                        {
                            tuple(subexperiment.parameters): np.array(
                                [
                                    parameters[self.param_map[param]]
                                    for param in subexperiment.parameters
                                ]
                            )
                        },
                    )
                    for subexperiment in subsystem
                ]
            )
            for label, subsystem in self.subexperiments.items()
        }
        results = {label: job.result() for label, job in jobs.items()}
        reconstructed_expval = reconstruct_expectation_values(
            cast(Dict[Hashable, PrimitiveResult | SamplerResult], results),
            self.exp_coefficients,
            self.subobservables,
        )
        return np.dot(reconstructed_expval, self.coefficients), list(results.values())


def create_estimate_cut(
    circuit: QuantumCircuit,
    observables: SparsePauliOp,
    cut_actions: List[CutAction],
    num_samples: int | float = np.inf,
):
    param_map = {param: i for i, param in enumerate(circuit.parameters)}
    circuit_w_ancilla = cut_circuit(circuit, cut_actions)
    observables_expanded = expand_observables(
        observables.paulis, circuit, circuit_w_ancilla
    )
    partitioned = partition_problem(
        circuit=circuit_w_ancilla, observables=observables_expanded
    )
    assert partitioned.subobservables is not None
    subexperiments, coefficients = generate_cutting_experiments(
        circuits=partitioned.subcircuits,
        observables=partitioned.subobservables,
        num_samples=num_samples,
    )
    assert isinstance(subexperiments, dict)
    return EstimateCut(
        subobservables=partitioned.subobservables,
        subexperiments=subexperiments,
        param_map=param_map,
        coefficients=observables.coeffs,
        exp_coefficients=coefficients,
    )


@dataclass
class SampleCut:
    subexperiments: Dict[Hashable, List[QuantumCircuit]]
    num_clbits: int
    param_map: Dict[Hashable, int]
    exp_coefficients: List[Tuple[float, WeightType]]

    def run(self, sampler: BaseSamplerV2, parameters: NDArray):
        for subsystem in self.subexperiments.values():
            for subexperiment in subsystem:
                print(subexperiment.parameters)
                print(
                    [
                        parameters[self.param_map[param]]
                        for param in subexperiment.parameters
                    ]
                )
                print(subexperiment.draw())

        jobs = {
            label: sampler.run(
                [
                    (
                        subexperiment,
                        {
                            tuple(subexperiment.parameters): np.array(
                                [
                                    parameters[self.param_map[param]]
                                    for param in subexperiment.parameters
                                ]
                            )
                        },
                    )
                    for subexperiment in subsystem
                ]
            )
            for label, subsystem in self.subexperiments.items()
        }
        results = {label: job.result() for label, job in jobs.items()}
        for result in results.values():
            for pub_result in result:
                print(pub_result.data.meas.get_counts())
        return reconstruct_distribution(
            cast(Dict[Hashable, PrimitiveResult | SamplerResult], results),
            self.num_clbits,
            self.exp_coefficients,
        )


def create_sample_cut(
    circuit: QuantumCircuit,
    cut_actions: List[CutAction],
    num_samples: int | float = np.inf,
):
    param_map = {param: i for i, param in enumerate(circuit.parameters)}
    measured_circuit = circuit.measure_all(inplace=False)
    assert measured_circuit is not None
    circuit_w_ancilla = cut_circuit(measured_circuit, cut_actions)
    partitioned = partition_problem(circuit=circuit_w_ancilla)
    subexperiments, coefficients = generate_cutting_experiments(
        circuits=partitioned.subcircuits,
        observables=None,
        num_samples=num_samples,
    )
    assert isinstance(subexperiments, dict)
    return SampleCut(
        subexperiments=subexperiments,
        num_clbits=measured_circuit.num_clbits,
        param_map=param_map,
        exp_coefficients=coefficients,
    )
