import numpy as np
from pathlib import Path
import pickle
import pytest
from summaries.scripts.compute_tree_summaries import __main__
from summaries.scripts.simulate_data import __main__ as __main__simulate_data


def test_compute_tree_summaries(tmp_path: Path) -> None:
    n = 23
    observed_path = tmp_path / "observed.pkl"
    output_path = tmp_path / "output.pkl"

    __main__simulate_data(
        [
            f"--n-samples={n}",
            "--n-observations=17",
            "TreeSimulationConfig",
            str(observed_path),
        ]
    )
    __main__([str(observed_path), str(output_path)])

    with observed_path.open("rb") as fp:
        observed = pickle.load(fp)

    with output_path.open("rb") as fp:
        output = pickle.load(fp)

    np.testing.assert_allclose(output["params"], observed["params"])
    assert output["data"].shape == (n, 5)

    pytest.shared.assert_pickle_loadable(output_path)
