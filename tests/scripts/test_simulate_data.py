from pathlib import Path
import pickle
import pytest
from summaries.scripts.simulate_data import __main__, SIMULATION_CONFIGS
from unittest import mock


@pytest.mark.parametrize("config", SIMULATION_CONFIGS)
def test_simulate_data(config: str, tmp_path: Path) -> None:
    output = tmp_path / "output.pkl"

    with mock.patch.object(SIMULATION_CONFIGS[config], "N_SAMPLES", 13):
        __main__([config, str(output)])

    with output.open("rb") as fp:
        result = pickle.load(fp)

    assert result["params"].ndim == 2
    assert result["params"].shape[0] == 13
