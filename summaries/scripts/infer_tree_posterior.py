from __future__ import annotations
import argparse
import networkx as nx
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from typing import List

from ..experiments.tree import expand_tree, TreeKernelPosterior
from .configs import TreeSimulationConfig


class InferTreePosteriorArgs:
    observed: Path
    output: Path


def __main__(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
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
