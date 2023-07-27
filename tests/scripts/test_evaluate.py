import numpy as np
import pandas as pd
from pathlib import Path
from summaries.scripts.evaluate import __main__
from summaries import util


def test_evaluate(tmp_path: Path) -> None:
    reference = np.random.normal(0, 1, (100, 10))
    paths = []
    for offset in [0, 3]:
        path = tmp_path / f"{offset}.pkl"
        util.dump_pickle({
            "params": reference,
            "samples": reference[:, None, :] + offset + np.random.normal(0, 1, (100, 1000, 10))
        }, path)
        paths.append(str(path))

    csv = tmp_path / "result.csv"
    __main__([f"--csv={csv}", *paths])

    result = pd.read_csv(csv).set_index("path")
    assert result.loc["0.pkl"].rmse < result.loc["3.pkl"].rmse
