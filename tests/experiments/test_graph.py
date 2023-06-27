import networkx as nx
from summaries.experiments.graph import compress_graph, expand_graph, simulate_graph


def test_compress_expand_graph() -> None:
    graph = simulate_graph(127, 0.5)

    predecessors = compress_graph(graph)
    assert predecessors.shape == (126,)
    # Node number 1 must have 0 as its predecessor.
    assert predecessors[0] == 0

    reconstructed = expand_graph(predecessors)
    assert nx.utils.graphs_equal(graph, reconstructed)
