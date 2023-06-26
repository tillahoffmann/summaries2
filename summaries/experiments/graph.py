import networkx as nx
import numpy as np
from typing import Tuple


def simulate_graph(n_nodes: int, gamma: int | None = None, seed: int | None = None) \
        -> Tuple[nx.DiGraph, float]:
    """
    Simulate a preferential attachment graph with power attachment kernel.

    Args:
        n_nodes: Number of nodes
        gamma: Power exponent.
        seed: Random number generator seed.

    Returns:
        Tuple of graph generated using a power attachment kernel and power exponent.
    """
    gamma = np.random.normal(0, 2) if gamma is None else gamma
    return nx.gn_graph(n_nodes, lambda k: k ** gamma, seed=seed), gamma


def compress_graph(graph: nx.DiGraph) -> np.ndarray:
    """
    Compress a growing-network digraph to an array of predecessors.

    Args:
        graph: Graph to compress.

    Returns:
        Vector of `n_nodes - 1` predecessors, starting with the predecessor of node `1`.
    """
    edges = np.asarray(list(graph.edges))
    idx = np.argsort(edges[:, 0])
    return edges[idx, 1]


def expand_graph(predecessors: np.ndarray) -> nx.DiGraph:
    """
    Expand an array of predecessors to a digraph.

    Args:
        predecessors: Array of predecessors, starting with the predecessor for node 1.

    Returns:
        Reconstructed digraph.
    """
    return nx.DiGraph(list(enumerate(predecessors, 1)))
