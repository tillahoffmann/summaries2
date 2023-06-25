import numpy as np
from pathlib import Path
import pickle
import pytest
from summaries.scripts.preprocess_coal import __main__ as __main__preprocess_coal
from summaries.scripts.train_transformer import __main__, TrainConfig, TRAIN_CONFIGS
from torch import as_tensor, nn, Tensor
from unittest import mock


@pytest.mark.parametrize("config", [x for x in TRAIN_CONFIGS if x.startswith("coal")])
def test_train_transformer_coal(config: str, tmp_path: Path) -> None:
    # Split up the data to test and training sets.
    coaloracle = Path(__file__).parent.parent / "data/coaloracle_sample.csv"
    __main__preprocess_coal(map(str, [coaloracle, tmp_path, "train:98", "validation:2"]))

    output = tmp_path / "output.pkl"
    argv = [config, tmp_path / "train.pkl", tmp_path / "validation.pkl", output]

    with mock.patch.object(TRAIN_CONFIGS[config], "max_epochs", 2):
        __main__(map(str, argv))

    with output.open("rb") as fp:
        result = pickle.load(fp)

    assert isinstance(result["transformer"], nn.Module)


class _Dummy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta = nn.Parameter(as_tensor(0.0))

    def forward(self, data: Tensor) -> Tensor:
        return 0 * self.theta * data[:, :2]


def test_train_stopping(tmp_path: Path) -> None:
    data_path = tmp_path / "data.pkl"
    with data_path.open("wb") as fp:
        pickle.dump({
            "data": np.random.normal(0, 1, (7, 3)),
            "params": np.random.normal(0, 1, (7, 2)),
        }, fp)

    output_path = tmp_path / "output.pkl"
    argv = ["test", data_path, data_path, output_path]

    train_config = TrainConfig(
        nn.MSELoss(),
        _Dummy,
    )

    with mock.patch.dict(TRAIN_CONFIGS, test=train_config):
        __main__(map(str, argv))

    assert output_path.is_file()
