from __future__ import annotations
import argparse
from fasttr import HistorySampler
import networkx as nx
import numpy as np
from pathlib import Path
import pickle
from scipy import integrate, optimize, special, stats
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from tqdm import tqdm
from typing import List

from ..experiments.graph import expand_graph
from .configs import GraphSimulationConfig


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


def __main__(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("observed", help="path to observed data", type=Path)
    parser.add_argument("output", help="path to output file", type=Path)
    args = parser.parse_args(argv)

    prior = GraphSimulationConfig.PRIOR

    # Use 101 elements just to catch accidental issues with tensor shapes.
    lin = np.linspace(prior.a, prior.b, 101)

    with args.observed.open("rb") as fp:
        observed = pickle.load(fp)
        n_observed = len(observed["data"])

    result = {}
    for gamma, predecessors in tqdm(zip(observed["params"], observed["data"]), total=n_observed):
        graph = nx.to_undirected(expand_graph(predecessors))
        posterior = TreeKernelPosterior(prior).fit(graph)
        log_prob = posterior.log_prob(lin)

        # Sanity check: Does the posterior integrate to roughly one?
        assert abs(integrate.trapz(np.exp(log_prob), lin) - 1) < 0.1

        result.setdefault("map_estimate", []).append(posterior.map_estimate_)
        result.setdefault("log_prob_actual", []).append(posterior.log_prob(gamma))
        result.setdefault("log_prob", []).append(log_prob)

    result = {key: np.asarray(value) for key, value in result.items()}
    result["map_estimate"] = result["map_estimate"].reshape((n_observed, 1))
    result["args"] = vars(args)
    result["lin"] = lin

    with open(args.output, "wb") as fp:
        pickle.dump(result, fp)


if __name__ == "__main__":
    __main__()
