from __future__ import annotations
import json
import numpy as np
from pathlib import Path
import pickle
import pytest
from sklearn.linear_model import LinearRegression
from summaries.scripts.infer import __main__, INFERENCE_CONFIGS, InferenceConfig
from summaries.scripts.preprocess_coal import __main__ as __main__preprocess_coal
from summaries.transformers import MinimumConditionalEntropyTransformer, NeuralTransformer, \
    PredictorTransformer, Transformer
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
        return PredictorTransformer(LinearRegression())


class DummyMinimumConditionalEntropyConfig(DummyConfig):
    IS_DATA_DEPENDENT = True

    def create_transformer(self, observed_data: Any | None = None) -> Transformer:
        return MinimumConditionalEntropyTransformer(observed_data, self.FRAC)


@pytest.mark.parametrize("config", [DummyPredictorConfig, DummyMinimumConditionalEntropyConfig])
def test_infer(simulated_data: np.ndarray, simulated_params: np.ndarray, observed_data: np.ndarray,
               tmp_path: Path, config: Type[DummyConfig]) -> None:
    # Create paths and write the data to disk.
    simulated = tmp_path / "simulated.pkl"
    observed = tmp_path / "observed.pkl"
    output = tmp_path / "output.pkl"

    with simulated.open("wb") as fp:
        pickle.dump({
            "data": simulated_data,
            "params": simulated_params,
        }, fp)

    with observed.open("wb") as fp:
        pickle.dump({
            "data": observed_data[:7],
        }, fp)

    with mock.patch.dict(INFERENCE_CONFIGS, test=config):
        __main__(map(str, ["test", simulated, observed, output]))

    with output.open("rb") as fp:
        result = pickle.load(fp)

    # Verify the shape of the sample.
    assert result["samples"].shape == (7, 1000, simulated_params.shape[-1])


@pytest.mark.parametrize("config_name", [x for x in INFERENCE_CONFIGS if x.startswith("Coal")])
def test_coal_infer(config_name: str, tmp_path: Path) -> None:
    # Split up the data to test and training sets.
    coaloracle = Path(__file__).parent.parent / "data/coaloracle_sample.csv"
    __main__preprocess_coal(map(str, [coaloracle, tmp_path, "simulated:98", "observed:2"]))

    # Prepare the arguments.
    output = tmp_path / "output.pkl"
    argv = [config_name, tmp_path / "simulated.pkl", tmp_path / "observed.pkl", output]

    # Dump a simple transformer if required.
    if config_name == "CoalescentNeuralConfig":
        transformer = tmp_path / "transformer.pkl"
        with transformer.open("wb") as fp:
            pickle.dump({
                "transformer": NeuralTransformer(nn.Linear(7, 2)),
            }, fp)
        argv.extend(["--transformer-kwargs", json.dumps({"transformer": str(transformer)})])

    # We only create a config here to
    config = INFERENCE_CONFIGS[config_name]

    # We need to increase the fraction of samples to estimate the entropy in this test.
    with mock.patch.object(config, "FRAC", 0.1):
        __main__(map(str, argv))

    with output.open("rb") as fp:
        result = pickle.load(fp)

    assert result["samples"].shape == (2, 9, 2)
