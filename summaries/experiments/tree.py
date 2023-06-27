from __future__ import annotations
from fasttr import HistorySampler
import networkx as nx
import numpy as np
from scipy import integrate, optimize, special, stats
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from typing import Any, Tuple


def simulate_tree(n_nodes: int, gamma: float, seed: Any | None = None) -> Tuple[nx.DiGraph, float]:
    """
    Simulate a preferential attachment tree with power attachment kernel.

    Args:
        n_nodes: Number of nodes
        gamma: Power exponent.
        seed: Random number generator seed.

    Returns:
        Tree generated using a power attachment kernel with exponent `gamma`.
    """
    return nx.gn_graph(n_nodes, lambda k: k ** gamma, seed=seed)


def compress_tree(tree: nx.DiGraph) -> np.ndarray:
    """
    Compress a growing tree to an array of predecessors.

    Args:
        tree: Tree to compress.

    Returns:
        Vector of `n_nodes - 1` predecessors, starting with the predecessor of node `1`.
    """
    edges = np.asarray(list(tree.edges))
    idx = np.argsort(edges[:, 0])
    return edges[idx, 1]


def expand_tree(predecessors: np.ndarray) -> nx.DiGraph:
    """
    Expand an array of predecessors to a tree.

    Args:
        predecessors: Array of predecessors, starting with the predecessor for node 1.

    Returns:
        Reconstructed tree.
    """
    return nx.DiGraph(list(enumerate(predecessors, 1)))


class TreeKernelPosterior(BaseEstimator):
    """
    Posterior for the kernel of a randomly grown tree (see
    https://doi.org/10.1103/PhysRevLett.126.038301 for details).

    Args:
        prior: Prior for the power attachment kernel exponent.
        n_history_samples: Number of histories to infer.
        n_samples: Number of posterior samples to draw using rejection sampling.

    Attributes:
        tree_: Tree the estimator was fit to.
        map_estimate_: Maximum a posteriori estimate of the power exponent.
    """
    def __init__(self, prior: stats.rv_continuous, *, n_history_samples: int = 100,
                 n_samples: int = 100) -> None:
        self.prior = prior
        self.n_history_samples = n_history_samples
        self.n_samples = n_samples

        self.tree_: nx.Tree | None = None
        self.map_estimate_: float | None = None

        self._sampler: HistorySampler | None = None
        self._log_norm: float | None = None
        self._max_log_density: float = 0

    def _log_target(self, gamma):
        """
        Evaluate the unnormalized log posterior. Once estimated, we subtract the maximum of the
        function for numericaly stability on exponentiation.
        """
        if not self._sampler:
            raise NotFittedError
        # Broadcast manually if gamma is an array.
        if isinstance(gamma, np.ndarray):
            return np.reshape([self._log_target(x) for x in np.ravel(gamma)], np.shape(gamma))
        self._sampler.set_kernel(kernel=lambda k: k ** gamma)
        log_likelihoods = self._sampler.get_log_posterior()
        log_prob = special.logsumexp(log_likelihoods) - np.log(self.n_history_samples) \
            + self.prior.logpdf(gamma) - self._max_log_density
        return log_prob

    def fit(self, tree: nx.Tree) -> TreeKernelPosterior:
        self.tree_ = tree
        self._sampler = HistorySampler(tree)
        self._sampler.sample(self.n_history_samples)

        # Find the maximum a posteriori estimate.
        result: optimize.OptimizeResult = optimize.minimize_scalar(
            lambda gamma: - self._log_target(gamma), method="bounded",
            bounds=[self.prior.a, self.prior.b],
        )
        assert result.success
        self.map_estimate_ = result.x
        self._max_log_density = - result.fun
        assert abs(self._log_target(self.map_estimate_)) < 1e-9

        # Integrate to find the normalization constant.
        norm, *_ = integrate.quad(lambda x: np.exp(self._log_target(x)), self.prior.a, self.prior.b)
        self._log_norm = np.log(norm)
        return self

    def log_prob(self, gamma):
        """
        Evaluate the log posterior.
        """
        if self._log_norm is None:
            raise NotFittedError
        return self._log_target(gamma) - self._log_norm

    def predict(self, data: nx.Graph) -> None:
        if not self.tree_:
            raise NotFittedError
        assert data is self.tree_, "Can only make predictions about the tree the estimator was " \
            "fit to."
        samples = []
        n_candidates = 2 * self.n_samples
        while len(samples) < self.n_samples:
            # Sample candidates uniformly from the domain of the prior.
            gamma = np.random.uniform(self.prior.a, self.prior.b, n_candidates)
            # Sample candidates uniformly at random. We don't need to do any scaling because, after
            # fitting, the maximum of `_log_target` is zero.
            p = np.log(np.random.uniform(0, 1, n_candidates) + 1e-12)
            # Accept samples where p is below the target.
            accept = p < self._log_target(gamma)
            samples.extend(gamma[accept])

            # Estimate the acceptance probability and scale up the rejection sample size
            # accordingly (up to an order of magnitude more than the samples we requested).
            remaining = self.n_samples - len(samples)
            accept_frac = max(np.mean(accept), 1 / self.n_samples)
            n_candidates = int(min(remaining / accept_frac, 10 * self.n_samples))

        # Only retain as many samples as requested and add an extra parameter dimension for
        # consistency with multi-dimensional parameters.
        return np.asarray(samples)[:self.n_samples, None]
