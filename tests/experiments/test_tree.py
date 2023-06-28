import networkx as nx
import numpy as np
import pytest
from scipy import stats
from sklearn.exceptions import NotFittedError
from summaries.experiments.tree import compress_tree, evaluate_gini, expand_tree, simulate_tree, \
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
    lin = np.linspace(*prior.support(), 200)
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


def test_evaluate_gini() -> None:
    x = np.ones(1000)
    np.testing.assert_allclose(evaluate_gini(x), 0)

    x[0] = 1e9
    assert evaluate_gini(x) > 0.99

    # Compare the Gini coefficients between a uniform and heavy-tailed distribution. The latter
    # should have a larger Gini.
    u = np.random.uniform(0, 1, 1000)
    h = np.exp(np.random.normal(0, 1, 1000))
    assert evaluate_gini(u) < evaluate_gini(h)
    assert 0 < evaluate_gini(u)
    assert 0 < evaluate_gini(h)

    # Should be invariant under rescaling.
    np.testing.assert_allclose(evaluate_gini(h), evaluate_gini(10 * h))
