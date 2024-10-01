from __future__ import annotations
from typing import Tuple, List, Union, Dict

from pydantic import BaseModel

import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives.containers.observables_array import (
    ObservablesArrayLike,
)

from .application import Application, ApplicationName, BitString

type InputNode = Union[int, str]
type EdgesInput = List[
    Union[Tuple[InputNode, InputNode, float], Tuple[InputNode, InputNode]]
]


class MaxcutOutput(BaseModel):
    """
    The output of the Maxcut application
    """

    s: List[InputNode]
    "Nodes in S"
    t: List[InputNode]
    "Nodes in T"
    edges: List[Tuple[InputNode, InputNode, float]]
    "The list of edges that need to be cut"

    def __hash__(self):
        return hash(self.model_dump_json())


class Maxcut(Application[EdgesInput, MaxcutOutput]):
    def __init__(self, edges: EdgesInput):
        """
        Create a new Maxcut application. This takes in a list of edges, where each edge is a tuple of two nodes and a weight. Nodes can be either strings or integers
        """
        self.edges, self.nodes = Maxcut._normalize_edges(edges)

    @classmethod
    def _normalize_edges(
        cls, input: EdgesInput
    ) -> Tuple[Dict[Tuple[int, int], float], List[InputNode]]:
        edges = []
        nodes = set()
        for edge in input:
            if len(edge) == 3:
                new_edge = edge
            elif len(edge) == 2:
                new_edge = (edge[0], edge[1], 1.0)
            else:
                raise ValueError("Invalid edge format")
            edges.append(new_edge)
            nodes.add(new_edge[0])
            nodes.add(new_edge[1])
        nodes = list(nodes)
        remap = {x: i for i, x in enumerate(nodes)}
        return (
            {(remap[e[0]], remap[e[1]]): e[2] for e in edges},
            nodes,
        )

    def name(self) -> ApplicationName:
        return "maxcut"

    def encode(self) -> EdgesInput:
        return [(self.nodes[i], self.nodes[j], w) for (i, j), w in self.edges.items()]

    @classmethod
    def decode(cls, encoded: EdgesInput) -> Maxcut:
        return cls(encoded)

    def hamiltonian(self) -> ObservablesArrayLike:
        pauli_list = []
        for (i, j), w in self.edges.items():
            paulis = ["I"] * len(self.nodes)
            paulis[i], paulis[j] = "Z", "Z"
            pauli_list.append(("".join(paulis)[::-1], w))
        return SparsePauliOp.from_list(pauli_list)

    def interpret(self, bit_string: BitString) -> MaxcutOutput:
        """
        Interpret the bit string as a solution to the Maxcut problem. This will return the nodes in S and T, as well as the edges that need to be cut
        """
        s = set()
        t = set()
        for i, bit in enumerate(np.flip(bit_string.to_list())):
            if bit:
                s.add(i)
            else:
                t.add(i)
        edges = []
        for u in s:
            for v in t:
                w = self.edges.get((u, v), self.edges.get((v, u)))
                if w is not None:
                    edges.append((self.nodes[u], self.nodes[v], w))
        s = sorted([self.nodes[i] for i in s])
        t = sorted([self.nodes[i] for i in t])
        return MaxcutOutput(s=s, t=t, edges=edges)
