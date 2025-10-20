import numpy as np
import pandas as pd
from pathlib import Path
import pytest
from summaries.scripts.evaluate import __main__
from summaries import util


@pytest.mark.parametrize("bounds", [None, "0,1,0,2,1,3"])
def test_evaluate(tmp_path: Path, bounds: str | None) -> None:
    reference = np.random.normal(0, 1, (100, 3))
    paths = []
    for offset in [0, 3]:
        path = tmp_path / f"{offset}.pkl"
        util.dump_pickle(
            {
                "params": reference,
                "samples": reference[:, None, :]
                + offset
                + np.random.normal(0, 1, (100, 1000, 3)),
            },
            path,
        )
        paths.append(str(path))

    csv = tmp_path / "result.csv"
    parts = [f"--csv={csv}"]
    if bounds:
        parts.append(f"--bounds={bounds}")

    __main__([*parts, *paths])

    result = pd.read_csv(csv).set_index("path")
    assert result.loc["0.pkl"].rmise < result.loc["3.pkl"].rmise
