import argparse
from pathlib import Path
import pickle
from torch import as_tensor, get_default_dtype, nn, no_grad
from torch.distributions import Distribution
import torch_geometric.data
import torch_geometric.loader
from typing import List

from ..experiments.tree import predecessors_to_datasets


class Args:
    n_samples: int
    mdn: Path
    observed: Path
    output: Path
    loader: str


def __main__(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser("infer_mdn")
    parser.add_argument("--n-samples", help="number of samples to draw", type=int, default=1000)
    parser.add_argument("--loader", choices={"raw", "tree"}, help="data loader", default="raw")
    parser.add_argument("mdn", help="path to trained mixture density network", type=Path)
    parser.add_argument("observed", help="path to observed data and parameters", type=Path)
    parser.add_argument("output", help="path to output file", type=Path)
    args: Args = parser.parse_args(argv)

    with args.mdn.open("rb") as fp:
        mdn: nn.Module = pickle.load(fp)["transformer"]

    with args.observed.open("rb") as fp:
        observed = pickle.load(fp)
        observed_data = observed["data"]

    if args.loader == "raw":
        observed_data = as_tensor(observed_data, dtype=get_default_dtype())
    elif args.loader == "tree":
        datasets = predecessors_to_datasets(observed_data)
        observed_data, = torch_geometric.loader.DataLoader(datasets, batch_size=len(datasets))
    else:
        raise NotImplementedError(args.loader)

    with no_grad():
        posterior: Distribution = mdn(observed_data)

    # This will have shape `(n_samples, batch_size, n_params)`, but the other samplers return
    # `(batch_size, n_samples, n_params)`. Let's move the axis.
    samples = posterior.sample([args.n_samples]).moveaxis(1, 0).numpy()

    with args.output.open("wb") as fp:
        pickle.dump({
            "args": vars(args),
            "samples": samples,
            "params": observed["params"]
        }, fp)


if __name__ == "__main__":
    __main__()
