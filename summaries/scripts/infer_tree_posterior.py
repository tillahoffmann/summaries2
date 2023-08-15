from __future__ import annotations
import argparse
import networkx as nx
import numpy as np
from pathlib import Path
import pickle
from scipy import stats
from snippets.empirical_distribution import sample_empirical_pdf
from tqdm import tqdm
from typing import List

from ..experiments.tree import expand_tree, TreeKernelPosterior
from .configs import TreeSimulationConfig


class InferTreePosteriorArgs:
    n_samples: int | None
    observed: Path
    output: Path


def __main__(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, help="number of posterior samples to draw")
    parser.add_argument("observed", help="path to observed data", type=Path)
    parser.add_argument("output", help="path to output file", type=Path)
    args: InferTreePosteriorArgs = parser.parse_args(argv)

    prior = TreeSimulationConfig.PRIOR

    # Evaluate a high-fidelity numerical posterior on the prior support so we can sample from it
    # approximately later.
    lin = np.linspace(*prior.support(), 500)

    with args.observed.open("rb") as fp:
        observed = pickle.load(fp)
        n_observed = len(observed["data"])

    result = {}
    for gamma, predecessors in tqdm(zip(observed["params"], observed["data"]), total=n_observed):
        tree = nx.to_undirected(expand_tree(predecessors))
        posterior = TreeKernelPosterior(prior).fit(tree)
        log_prob = posterior.log_prob(lin)

        if args.n_samples:
            # Sample from the empirical pdf with a small tolerance for renormalizing the CDF. This
            # is necessary in rare cases because the normalization evaluated using `quad` in the
            # `fit` method of `TreeKernelPosterior` may differ numerically from the `trapz`
            # integration used by `sample_empirical_pdf`.
            samples = sample_empirical_pdf(lin, np.exp(log_prob), args.n_samples, tol=0.1)
            # Store and add trailing dimension for consistency with multivariate problems.
            result.setdefault("samples", []).append(samples[..., None])

        result.setdefault("map_estimate", []).append(posterior.map_estimate_)
        result.setdefault("log_prob_actual", []).append(posterior.log_prob(gamma))
        result.setdefault("log_prob", []).append(log_prob)

        # Record the Spearman rank correlation between inferred and actual history.
        histories = np.asarray(posterior._sampler.get_histories())
        spearman = stats.spearmanr(np.arange(histories.shape[1]), histories.mean(axis=0))
        result.setdefault("spearman", []).append(spearman.statistic)

    result = {key: np.asarray(value) for key, value in result.items()}
    result["map_estimate"] = result["map_estimate"].reshape((n_observed, 1))
    result["args"] = vars(args)
    result["lin"] = lin
    result["params"] = observed["params"]

    with open(args.output, "wb") as fp:
        pickle.dump(result, fp)


if __name__ == "__main__":
    __main__()
