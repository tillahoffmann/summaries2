from __future__ import annotations
import json
import numpy as np
from pathlib import Path
import pickle
import pytest
from sklearn.linear_model import LinearRegression
from summaries.scripts.infer_posterior import __main__, INFERENCE_CONFIGS, InferenceConfig
from summaries.scripts.preprocess_coalescent import __main__ as __main__preprocess_coalescent
from summaries.scripts.simulate_data import __main__ as __main__simulate_data
from summaries.scripts.configs import TreeSimulationConfig
from summaries.transformers import as_transformer, MinimumConditionalEntropyTransformer, \
    NeuralTransformer, Transformer
from torch import nn
from typing import Any, Type
from unittest import mock


class DummyPreprocessor:
    def fit(self, *args) -> DummyPreprocessor:
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        # Remove the last feature as a "preprocessing step".
        return data[:, :-1]


class DummyConfig(InferenceConfig):
    FRAC = 0.01

    def create_preprocessor(self) -> Transformer | None:
        return DummyPreprocessor()


class DummyPredictorConfig(DummyConfig):
    def create_transformer(self, observed_data: Any | None = None) -> Transformer:
        return as_transformer(LinearRegression)()


class DummyMinimumConditionalEntropyConfig(DummyConfig):
    IS_DATA_DEPENDENT = True

    def create_transformer(self, observed_data: Any | None = None) -> Transformer:
        return MinimumConditionalEntropyTransformer(observed_data, self.FRAC)


@pytest.mark.parametrize("config", [DummyPredictorConfig, DummyMinimumConditionalEntropyConfig])
def test_infer(simulated_data: np.ndarray, simulated_params: np.ndarray, observed_data: np.ndarray,
               tmp_path: Path, config: Type[DummyConfig]) -> None:
    # Create paths and write the data to disk.
    simulated_path = tmp_path / "simulated.pkl"
    observed_path = tmp_path / "observed.pkl"
    output_path = tmp_path / "output.pkl"

    with simulated_path.open("wb") as fp:
        pickle.dump({
            "data": simulated_data,
            "params": simulated_params,
        }, fp)

    with observed_path.open("wb") as fp:
        pickle.dump({
            "data": observed_data[:7],
        }, fp)

    with mock.patch.dict(INFERENCE_CONFIGS, test=config):
        __main__(map(str, ["test", simulated_path, observed_path, output_path]))

    with output_path.open("rb") as fp:
        output = pickle.load(fp)

    # Verify the shape of the sample.
    assert output["samples"].shape == (7, 1000, simulated_params.shape[-1])

    pytest.shared.check_pickle_loadable(output_path)


@pytest.mark.parametrize("config", [x for x in INFERENCE_CONFIGS if x.startswith("Coalescent")])
def test_coalescent_infer(config: str, tmp_path: Path) -> None:
    # Split up the data to test and training sets.
    coaloracle = Path(__file__).parent.parent / "data/coaloracle_sample.csv"
    __main__preprocess_coalescent(map(str, [coaloracle, tmp_path, "simulated:98", "observed:2"]))

    # Prepare the arguments.
    output_path = tmp_path / "output.pkl"
    argv = [config, tmp_path / "simulated.pkl", tmp_path / "observed.pkl", output_path]

    # Dump a simple transformer if required.
    if config == "CoalescentNeuralConfig":
        transformer = tmp_path / "transformer.pkl"
        with transformer.open("wb") as fp:
            pickle.dump({
                "transformer": NeuralTransformer(nn.Linear(7, 2)),
            }, fp)
        argv.extend(["--transformer-kwargs", json.dumps({"transformer": str(transformer)})])

    # We need to increase the fraction of samples to estimate the entropy in this test.
    with mock.patch.object(INFERENCE_CONFIGS[config], "FRAC", 0.1):
        __main__(map(str, argv))

    with output_path.open("rb") as fp:
        output = pickle.load(fp)

    assert output["samples"].shape == (2, 9, 2)

    pytest.shared.check_pickle_loadable(output_path)


@pytest.mark.parametrize("config", [x for x in INFERENCE_CONFIGS if x.startswith("Tree")])
def test_tree_infer(config: str, tmp_path: Path) -> None:
    # Generate some data.
    simulated_path = tmp_path / "simulated.pkl"
    observed_path = tmp_path / "observed.pkl"

    with mock.patch.object(TreeSimulationConfig, "N_NODES", 17):
        __main__simulate_data(["--n-samples=37", "TreeSimulationConfig", str(simulated_path)])
        __main__simulate_data(["--n-samples=5", "TreeSimulationConfig", str(observed_path)])

    output_path = tmp_path / "output.pkl"
    argv = [config, tmp_path / "simulated.pkl", tmp_path / "observed.pkl", output_path]

    with mock.patch.object(INFERENCE_CONFIGS[config], "FRAC", 0.1):
        __main__(map(str, argv))

    with output_path.open("rb") as fp:
        output = pickle.load(fp)

    assert output["samples"].shape == (5, 3, 1)

    pytest.shared.check_pickle_loadable(output_path)
