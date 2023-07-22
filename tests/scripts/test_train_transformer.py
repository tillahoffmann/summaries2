import numpy as np
from pathlib import Path
import pickle
import pytest
from summaries.scripts.preprocess_coalescent import __main__ as __main__preprocess_coalescent
from summaries.scripts.simulate_data import __main__ as __main__simulate_data
from summaries.scripts.train_transformer import __main__, TrainConfig, TRAIN_CONFIGS
from torch import as_tensor, nn, Tensor
from unittest import mock


@pytest.mark.parametrize("config", [x for x in TRAIN_CONFIGS if x.startswith("Coalescent")])
def test_train_transformer_coalescent(config: str, tmp_path: Path) -> None:
    # Split up the data to test and training sets.
    coaloracle = Path(__file__).parent.parent / "data/coaloracle_sample.csv"
    __main__preprocess_coalescent(map(str, [coaloracle, tmp_path, "train:98", "validation:2"]))

    output = tmp_path / "output.pkl"
    argv = [config, tmp_path / "train.pkl", tmp_path / "validation.pkl", output]

    with mock.patch.object(TRAIN_CONFIGS[config], "MAX_EPOCHS", 2):
        __main__(map(str, argv))

    with output.open("rb") as fp:
        result = pickle.load(fp)

    assert isinstance(result["transformer"], nn.Module)


@pytest.mark.parametrize("config", [x for x in TRAIN_CONFIGS if x.startswith("Tree")])
def test_train_transformer_tree(config: str, tmp_path: Path) -> None:
    # Generate two datasets.
    train_path = tmp_path / "train.pkl"
    validation_path = tmp_path / "validation.pkl"
    for path in [train_path, validation_path]:
        __main__simulate_data(["TreeSimulationConfig", "--n-observations=13", "--n-samples=10",
                               str(path)])

    output_path = tmp_path / "output.pkl"
    argv = [config, tmp_path / "train.pkl", tmp_path / "validation.pkl", output_path]

    with mock.patch.object(TRAIN_CONFIGS[config], "MAX_EPOCHS", 2):
        __main__(map(str, argv))

    pytest.shared.assert_pickle_loadable(output_path)
    with output_path.open("rb") as fp:
        result = pickle.load(fp)

    assert isinstance(result["transformer"], nn.Module)


# Outside the test function to support pickling in tests.
class DummyModule(nn.Module):
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

    class DummyConfig(TrainConfig):
        LOSS = nn.MSELoss()

        def create_transformer(self):
            return DummyModule()

    with mock.patch.dict(TRAIN_CONFIGS, test=DummyConfig):
        __main__(map(str, argv))

    assert output_path.is_file()
