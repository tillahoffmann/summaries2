import networkx as nx
import numpy as np
import pytest
from scipy import stats
from sklearn.exceptions import NotFittedError
from summaries.experiments.graph import compress_graph, expand_graph, simulate_graph, \
    TreeKernelPosterior


def test_compress_expand_graph() -> None:
    graph = simulate_graph(127, 0.5)

    predecessors = compress_graph(graph)
    assert predecessors.shape == (126,)
    # Node number 1 must have 0 as its predecessor.
    assert predecessors[0] == 0

    reconstructed = expand_graph(predecessors)
    assert nx.utils.graphs_equal(graph, reconstructed)


def test_tree_posterior() -> None:
    graph = simulate_graph(127, 0.5)
    prior = stats.uniform(0.2, 0.9)
    posterior = TreeKernelPosterior(prior)
    posterior.fit(graph)

    # Check normalization of posterior.
    lin = np.linspace(prior.a, prior.b, 200)
    log_prob = posterior.log_prob(lin)
    assert (np.trapz(np.exp(log_prob), lin) - 1) < 0.01


def test_tree_posterior_not_fitted() -> None:
    with pytest.raises(NotFittedError):
        TreeKernelPosterior(stats.uniform(0, 1))._log_target(0.5)
    with pytest.raises(NotFittedError):
        TreeKernelPosterior(stats.uniform(0, 1)).log_prob(0.5)
