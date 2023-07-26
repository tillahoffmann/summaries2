from pathlib import Path
import pickle
import pytest
from summaries.scripts.simulate_data import __main__, SIMULATION_CONFIGS


@pytest.mark.parametrize("config", SIMULATION_CONFIGS)
def test_simulate_data(config: str, tmp_path: Path) -> None:
    output = tmp_path / "output.pkl"

    __main__([config, "--n-observations=17", "--n-samples=13", str(output)])

    with output.open("rb") as fp:
        result = pickle.load(fp)

    assert result["params"].ndim == 2
    assert result["params"].shape[0] == 13

    if config == "BenchmarkSimulationConfig":
        assert result["params"].shape == (13, 1)
        assert result["data"].shape == (13, 17, 2)
