import argparse
from pathlib import Path
import pickle
from torch import as_tensor, get_default_dtype, nn, no_grad
from torch.distributions import Distribution
from typing import List


class Args:
    n_samples: int
    mdn: Path
    observed: Path
    output: Path


def __main__(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser("infer_mdn")
    parser.add_argument("--n-samples", help="number of samples to draw", type=int, default=1000)
    parser.add_argument("mdn", help="path to trained mixture density network", type=Path)
    parser.add_argument("observed", help="path to observed data and parameters", type=Path)
    parser.add_argument("output", help="path to output file", type=Path)
    args: Args = parser.parse_args(argv)

    with args.mdn.open("rb") as fp:
        mdn: nn.Module = pickle.load(fp)["transformer"]

    with args.observed.open("rb") as fp:
        observed = pickle.load(fp)["data"]

    # Cast to tensor; this will need to be modified for data that are not simply arrays.
    observed = as_tensor(observed, dtype=get_default_dtype())

    with no_grad():
        posterior: Distribution = mdn(observed)

    # This will have shape `(n_samples, batch_size, n_params)`, but the other samplers return
    # `(batch_size, n_samples, n_params)`. Let's move the axis.
    samples = posterior.sample([args.n_samples]).moveaxis(1, 0).numpy()

    with args.output.open("wb") as fp:
        pickle.dump({
            "args": vars(args),
            "samples": samples,
        }, fp)


if __name__ == "__main__":
    __main__()
