from __future__ import annotations
from fasttr import HistorySampler
import networkx as nx
import numpy as np
from scipy import integrate, optimize, special, stats
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from typing import Any, Tuple


def simulate_graph(n_nodes: int, gamma: float, seed: Any | None = None) -> Tuple[nx.DiGraph, float]:
    """
    Simulate a preferential attachment graph with power attachment kernel.

    Args:
        n_nodes: Number of nodes
        gamma: Power exponent.
        seed: Random number generator seed.

    Returns:
        Graph generated using a power attachment kernel with exponent `gamma`.
    """
    return nx.gn_graph(n_nodes, lambda k: k ** gamma, seed=seed)


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


class TreeKernelPosterior(BaseEstimator):
    """
    Posterior for the kernel of a randomly grown tree (see
    https://doi.org/10.1103/PhysRevLett.126.038301 for details).

    Args:
        n_history_samples: Number of histories to infer.

    Attributes:
        graph_: Graph the estimator was fit to.
        map_estimate_: Maximum a posteriori estimate of the power exponent.
    """
    def __init__(self, prior: stats.rv_continuous, n_history_samples: int = 100):
        self.prior = prior
        self.n_history_samples = n_history_samples

        self.graph_: nx.Graph | None = None
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
        self._sampler.set_kernel(kernel=lambda k: k ** gamma)
        log_likelihoods = self._sampler.get_log_posterior()
        log_prob = special.logsumexp(log_likelihoods) - np.log(self.n_history_samples) \
            + self.prior.logpdf(gamma) - self._max_log_density
        return log_prob

    def fit(self, graph: nx.Graph) -> TreeKernelPosterior:
        self.graph_ = graph
        self._sampler = HistorySampler(graph)
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
        if isinstance(gamma, float):
            return self._log_target(gamma) - self._log_norm
        return np.reshape([self.log_prob(x) for x in np.ravel(gamma)], np.shape(gamma))
