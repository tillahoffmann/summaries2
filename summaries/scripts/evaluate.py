import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from snippets.stats import GaussianKernelDensity
from typing import List

from ..util import load_pickle


class Args:
    bounds: np.ndarray | None
    sigfigs: int
    observed: Path
    results: List[Path]


def _parse_bounds(bounds: str) -> np.ndarray:
    return None if bounds == "none" else np.reshape([float(x) for x in bounds.split(",")], (-1, 2))


def __main__(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bounds", type=_parse_bounds, help="bounds for kernel density estimation")
    parser.add_argument("--sigfigs", type=int, help="number of significant figures for statistics")
    parser.add_argument("--csv", type=Path, help="path to CSV output file")
    parser.add_argument("results", type=Path, nargs="+", help="path to posterior samples")
    args: Args = parser.parse_args(argv)

    statistics = []
    reference_params = None
    for path in args.results:
        result = load_pickle(path)
        if reference_params is None:
            reference_params = result["params"]
        else:
            np.testing.assert_allclose(result["params"], reference_params)

        observed_params = reference_params  # Shape (n_examples, n_params).
        samples = result["samples"]  # Shape (n_examples, n_samples, n_params).
        n_examples = samples.shape[0]
        err_factor = 1 / np.sqrt(n_examples - 1)

        # Evaluate the root mean squared error.
        rmses = np.sqrt(np.square(observed_params[:, None, :] - samples).sum(2).mean(1))
        rmse = np.mean(rmses)
        rmse_err = np.std(rmses) * err_factor

        # Evaluate the negative log probability using a kernel density estimator.
        nlps = []
        for x, xs in zip(observed_params, samples):
            estimator = GaussianKernelDensity(bounds=args.bounds).fit(xs)
            nlp, = - estimator.score_samples([x])
            nlps.append(nlp)
        nlps = np.asarray(nlps)
        assert nlps.shape == (n_examples,)
        nlp = np.mean(nlps)
        nlp_err = np.std(nlps) * err_factor

        statistics.append({
            "path": path.name,
            "nlp": nlp,
            "nlp_err": nlp_err,
            "rmse": rmse,
            "rmse_err": rmse_err,
        })
    statistics = pd.DataFrame(statistics).sort_values("nlp")
    if args.csv:
        statistics.to_csv(args.csv, index=False)

    for key in ["rmse", "nlp"]:
        sigfigs = args.sigfigs or int(np.ceil(-np.log10(statistics[f"{key}_err"])).max())
        statistics[f"{key}_err"] = statistics[f"{key}_err"].round(sigfigs)
        statistics[key] = statistics[key].round(sigfigs)

    pd.set_option('display.max_colwidth', None)
    print(statistics)


if __name__ == "__main__":
    __main__()
