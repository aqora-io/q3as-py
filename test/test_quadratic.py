from q3as.algo.vqe import VQE
from q3as.quadratic import QuadraticProgram, QuadraticProgramToQubo

import numpy as np

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator, SamplerV2 as Sampler

from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz, TwoLocal, EfficientSU2
from qiskit.primitives.containers.bindings_array import BindingsArray

simulator = AerSimulator()


def test_quadratic():
    qp = QuadraticProgram("qp")

    qp.integer_var(name="x", lowerbound=-1, upperbound=1)
    qp.integer_var(name="y", lowerbound=-1, upperbound=5)
    qp.integer_var(name="z", lowerbound=-1, upperbound=5)
    qp.minimize(constant=3, linear={"x": 1}, quadratic={("x", "y"): 2, ("z", "z"): -1})

    print(qp.prettyprint())

    converter = QuadraticProgramToQubo()
    converted = converter.convert(qp)
    print(converted.prettyprint())
    op, offset = converter.convert(qp).to_ising()
    print("offset =", offset)

    # ansatz = transpile(QAOAAnsatz(op), simulator)
    ansatz = transpile(EfficientSU2(op.num_qubits), simulator)

    estimator = Estimator(options={"run_options": {"default_shots": 1024}})
    result = VQE.builder().observables(op).ansatz(ansatz).run(estimator)
    if result.estimated is None:
        raise ValueError("Expected a result")
    sampler = Sampler(default_shots=1024)
    bound_params = BindingsArray.coerce(
        {tuple(ansatz.parameters): result.estimated.params}
    )
    results = sampler.run([(ansatz.measure_all(inplace=False), bound_params)]).result()
    counts = results[0].data.meas.get_counts()
    # TODO check if empty
    best = max([(v, k) for k, v in counts.items()])[1]
    bits = [b == "1" for b in best]
    bits.reverse()
    interpreted = converter.interpret(bits)
    for name, index in qp.variables_index.items():
        print(f"{name} = {interpreted[index]}")
    print(interpreted)
    print("result = ", qp.objective.evaluate(interpreted))
    print("result2 = ", qp.objective.evaluate([-1.0, 5.0, 5.0]))
