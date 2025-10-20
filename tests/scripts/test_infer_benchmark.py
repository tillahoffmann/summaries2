from pathlib import Path
import pickle
import pytest

from summaries.scripts.infer_benchmark import __main__
from summaries.scripts.simulate_data import __main__ as __main__simulate_data


def test_benchmark_infer(tmp_path: Path) -> None:
    observed_path = tmp_path / "observed.pkl"
    __main__simulate_data(
        map(str, ["--n-samples=7", "BenchmarkSimulationConfig", observed_path])
    )

    # Prepare the arguments.
    output_path = tmp_path / "output.pkl"
    argv = ["--n-samples=9", observed_path, output_path]

    __main__(list(map(str, argv)))

    with output_path.open("rb") as fp:
        output = pickle.load(fp)

    assert output["samples"].shape == (7, 9, 1)
    pytest.shared.assert_pickle_loadable(output_path)
