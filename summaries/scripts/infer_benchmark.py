import argparse
import cmdstanpy
import logging
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from typing import List


class Args:
    n_samples: int
    observed: Path
    output: Path


def __main__(argv: List[str] | None = None) -> None:
    # Disable verbose cmdstan logging.
    logger = cmdstanpy.utils.get_logger()
    logger.setLevel(logging.ERROR)
    for handler in logger.handlers:
        handler.setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, help="number of posterior samples", default=1000)
    parser.add_argument("observed", type=Path, help="observed data path")
    parser.add_argument("output", type=Path, help="output path")
    args: Args = parser.parse_args(argv)

    with open(args.observed, "rb") as fp:
        observed = pickle.load(fp)

    chains = 4
    model = cmdstanpy.CmdStanModel(stan_file=Path(__file__).parent / "infer_benchmark.stan")

    samples = []
    for i, x in enumerate(tqdm(observed["data"].squeeze())):
        fit = model.sample({"x": x, "n_observations": x.size, "variance_offset": 1}, chains=chains,
                           iter_sampling=args.n_samples, show_progress=False, adapt_delta=0.97)
        diagnosis = fit.diagnose()
        if "no problems detected" not in diagnosis:
            print(f"Problems detected for sample with index {i}:\n{diagnosis}")
        samples.append(fit.theta[::chains, None])

    with open(args.output, "wb") as fp:
        pickle.dump({
            "args": vars(args),
            "samples": np.asarray(samples),
            "params": observed["params"],
        }, fp)


if __name__ == "__main__":
    __main__()
