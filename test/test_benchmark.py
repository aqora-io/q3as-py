import pytest
import os
import random

from q3as.algo.vqe import VQE
from q3as.app import Maxcut
from q3as import Client, Credentials

from qiskit_aer import AerSimulator

simulator = AerSimulator()

benchmark = pytest.mark.skipif("not config.getoption('benchmark')")

url = os.getenv("Q3AS_URL", "http://localhost:8080")


def gen_graph(n, p=0.5, weight_range=(1, 1)):
    """
    Generates a random graph using the Erdős–Rényi model (G(n, p)).

    Parameters:
        n (int): Number of nodes.
        p (float): Probability of adding an edge between any two nodes.
        weight_range (tuple): Range (min, max) for the random weights of edges.

    Returns:
        edges (list of tuples): A list of edges represented as (node1, node2, weight).
    """
    edges = []

    # Iterate through all pairs of nodes (i, j) where i < j
    for i in range(n):
        for j in range(i + 1, n):
            # Include an edge with probability `p`
            if random.random() < p:
                # Generate a random weight within the specified range
                weight = random.randint(weight_range[0], weight_range[1])
                edges.append((i, j, weight))

    return edges


@benchmark
def test_benchmark():
    with Client(Credentials.load(".credentials.json"), url=url) as client:
        for size in [15, 20, 25, 28, 30, 32]:
            print("Running Maxcut with size", size)
            job = VQE.builder().app(Maxcut(gen_graph(size))).send(client)
            print("Started job with name:", job.name)
            print("Result", job.result())
