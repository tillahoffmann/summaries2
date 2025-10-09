import argparse
import networkit as nk
import numpy as np
from tqdm import tqdm
from typing import List

from ..experiments.tree import evaluate_gini, expand_tree
from ..util import dump_pickle, load_pickle


def __main__(argv: List[str] | None = None) -> None:
    """
    Precompute expert summary statistics for inferring the attachment kernel of trees.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args(argv)

    observed = load_pickle(args.input)
    observed_data = observed["data"]

    summaries = []
    for predecessors in tqdm(observed_data, desc="evaluate tree kernel summary candidates"):
        tree = nk.graphtools.toUndirected(expand_tree(predecessors, use="networkit"))
        in_degrees = np.bincount(predecessors, minlength=observed_data.shape[-1] + 1)
        betweenness = nk.centrality.Betweenness(tree)
        betweenness.run()
        diameter = nk.distance.Diameter(tree)
        diameter.run()
        summaries.append((
            in_degrees.std(),
            evaluate_gini(in_degrees),
            max(betweenness.scores()),
            diameter.getDiameter()[0],
            np.random.uniform(0, 1),
        ))
    summaries = np.asarray(summaries)

    dump_pickle(observed | {"data": summaries}, args.output)


if __name__ == "__main__":
    __main__()
