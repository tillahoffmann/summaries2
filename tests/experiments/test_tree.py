import networkx as nx
import numpy as np
import pytest
from scipy import stats
from sklearn.exceptions import NotFittedError
from summaries.experiments.tree import compress_tree, expand_tree, simulate_tree, \
    TreeKernelPosterior


def test_compress_expand_tree() -> None:
    tree = simulate_tree(127, 0.5)

    predecessors = compress_tree(tree)
    assert predecessors.shape == (126,)
    # Node number 1 must have 0 as its predecessor.
    assert predecessors[0] == 0

    reconstructed = expand_tree(predecessors)
    assert nx.utils.graphs_equal(tree, reconstructed)


def test_tree_posterior() -> None:
    tree = simulate_tree(127, 0.5)
    prior = stats.uniform(0.2, 0.9)
    posterior = TreeKernelPosterior(prior, n_samples=57)
    posterior.fit(tree)

    # Check normalization of the posterior.
    lin = np.linspace(prior.a, prior.b, 200)
    log_prob = posterior.log_prob(lin)
    assert (np.trapz(np.exp(log_prob), lin) - 1) < 0.01

    # Draw a few samples using rejection sampling.
    assert posterior.predict(tree).shape == (57, 1)


def test_tree_posterior_not_fitted() -> None:
    with pytest.raises(NotFittedError):
        TreeKernelPosterior(stats.uniform(0, 1)).predict(None)
    with pytest.raises(NotFittedError):
        TreeKernelPosterior(stats.uniform(0, 1))._log_target(0.5)
    with pytest.raises(NotFittedError):
        TreeKernelPosterior(stats.uniform(0, 1)).log_prob(0.5)
