from __future__ import annotations
from fasttr import HistorySampler
import networkx as nx
import numpy as np
from scipy import integrate, optimize, special, stats
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
import torch
from torch import nn, Tensor
from torch.distributions import AffineTransform, Beta, Categorical, Independent, \
    MixtureSameFamily, TransformedDistribution
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
from torch_geometric.utils import to_undirected
from typing import Any, List, Tuple

from ..nn import MeanPoolByGraph, SequentialWithKeywords
from ..transformers import NeuralTransformer


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
            bounds=self.prior.support(),
        )
        assert result.success
        self.map_estimate_ = result.x
        self._max_log_density = - result.fun
        assert abs(self._log_target(self.map_estimate_)) < 1e-9

        # Integrate to find the normalization constant.
        norm, *_ = integrate.quad(lambda x: np.exp(self._log_target(x)), *self.prior.support())
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
            gamma = np.random.uniform(*self.prior.support(), n_candidates)
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


def evaluate_gini(x: np.ndarray) -> float:
    """
    Evaluate the Gini coefficient based on the last expression of
    https://en.wikipedia.org/wiki/Gini_coefficient#Alternative_expressions.

    Args:
        x: Sample from the population.

    Returns:
        Gini coefficient.
    """
    n = x.size
    x = np.sort(x)
    return 1 - 2 / (n - 1) * (n - (1 + np.arange(n)) @ x / x.sum())


class _TreeTransformer(NeuralTransformer):
    """
    Learnable transformer for the tree kernel inference problem without the "head" of the network.
    """
    def __init__(self, depth: int = 2) -> None:
        layers = []
        for _ in range(depth):
            layers.append(GINConv(nn.Sequential(
                nn.LazyLinear(8),
                nn.Tanh(),
                nn.LazyLinear(8),
                nn.Tanh(),
            )))
        layers.extend([nn.LazyLinear(1), nn.Tanh(), MeanPoolByGraph()])
        transformer = SequentialWithKeywords(*layers)
        super().__init__(transformer)


class TreePosteriorMixtureDensityTransformer(_TreeTransformer):
    """
    Learnable transformer with conditional posterior density estimation "head" based on mixture
    density networks.
    """
    def __init__(self, n_components: int = 10) -> None:
        super().__init__()
        self.n_components = n_components
        self.mixture_parameters = nn.ModuleDict({
            key: nn.Sequential(
                nn.LazyLinear(8),
                nn.Tanh(),
                nn.LazyLinear(size * n_components),
            ) for (size, key) in [(1, "logits"), (1, "concentration1s"), (1, "concentration0s")]
        })

    def transform(self, data: Data) -> torch.Tensor:
        features = torch.ones([data.num_nodes, 1])
        return self.transformer(features, edge_index=data.edge_index, batch=data.batch)

    def forward(self, data: Data) -> Tensor:
        transformed = self.transform(data)

        logits: Tensor = self.mixture_parameters["logits"](transformed)
        mixture_dist = Categorical(logits=logits)

        concentration1s: Tensor = self.mixture_parameters["concentration1s"](transformed)
        concentration1s = concentration1s.reshape((-1, self.n_components, 1)).exp()
        concentration0s: Tensor = self.mixture_parameters["concentration0s"](transformed)
        concentration0s = concentration0s.reshape((-1, self.n_components, 1)).exp()
        component_dist = Beta(concentration1s, concentration0s)

        # Rescale the distribution to the domain used by the simulations and reinterpret the last
        # batch dimension as an event dimension.
        component_dist = TransformedDistribution(component_dist, AffineTransform(0, 2))
        component_dist = Independent(component_dist, 1)

        return MixtureSameFamily(mixture_dist, component_dist)


def predecessors_to_datasets(predecessors: np.ndarray, params: np.ndarray | None = None,
                             device: torch.device | None = None) -> List[Data]:
    """
    Convert a matrix of predecessors to a list of `torch_geometric` datasets.
    """
    datasets = []
    for i, row in enumerate(predecessors):
        n_edges, = row.shape
        edge_index = torch.vstack([
            1 + torch.arange(n_edges, device=device)[None],
            torch.as_tensor(row[None], dtype=torch.int64, device=device),
        ])
        edge_index = to_undirected(edge_index)

        kwargs = {}
        if params is not None:
            kwargs["params"] = torch.as_tensor(params[i, None], dtype=torch.get_default_dtype())

        datasets.append(Data(edge_index=edge_index, num_nodes=n_edges + 1, **kwargs))
    return datasets
