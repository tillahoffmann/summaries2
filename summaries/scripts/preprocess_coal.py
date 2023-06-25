from argparse import ArgumentParser
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import List, Optional, Tuple

from .base import resolve_path


def _parse_splits(value: str) -> Tuple[Path, int]:
    path, size = value.split(":")
    return Path(path), int(size)


class Args:
    coaloracle: Path
    output: Path
    splits: List[Tuple[Path, int]]


def __main__(argv: Optional[List[str]] = None) -> None:
    parser = ArgumentParser("preprocess_coal")
    parser.add_argument("--seed", help="random number generator seed", type=np.random.seed)
    parser.add_argument("coaloracle", help="path to file containing coalescent samples")
    parser.add_argument("output", help="output directory path", type=resolve_path)
    parser.add_argument("splits", help="path to dataset splits and sizes as `[path]:[size]`",
                        type=_parse_splits, nargs="+")
    args: Args = parser.parse_args(argv)

    # Load the data and shuffle it; verify that the first two columns are parameters.
    data = pd.read_csv(args.coaloracle).sample(frac=1, replace=False)
    columns = list(data.columns)
    assert columns[:2] == ["theta", "rho"]

    # Verify the split sizes are correct.
    splits = dict(args.splits)
    assert (size := sum(splits.values())) == data.shape[0], f"Splits {splits} must add up to " \
        f"dataset size {data.shape[0]}; got {size}."

    # Split up the data and dump the results in the format expected by the `infer` script.
    offset = 0
    for split, size in splits.items():
        with (args.output / f"{split}.pkl").open("wb") as fp:
            pickle.dump({
                "args": vars(args),
                "split": split,
                "size": size,
                "data_columns": columns[2:],
                "data": data.values[offset:offset + size, 2:],
                "param_columns": columns[:2],
                "params": data.values[offset:offset + size, :2],
            }, fp)
        offset += size


if __name__ == "__main__":
    __main__()
