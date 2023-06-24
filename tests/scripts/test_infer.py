from __future__ import annotations
import numpy as np
from pathlib import Path
import pickle
import pytest
from sklearn.linear_model import LinearRegression
from summaries.scripts.infer import __main__, INFERENCE_CONFIGS, InferenceConfig
from summaries.transformers import as_transformer, MinimumConditionalEntropyTransformer, Transformer
from typing import Any, Dict, Type
from unittest import mock


class TestPreprocessor:
    def fit(self, *args) -> TestPreprocessor:
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        # Remove the last feature as a "preprocessing step".
        return data[:, :-1]


@pytest.mark.parametrize("transformer_cls, transformer_kwargs", [
    (as_transformer(LinearRegression), {}),
    (MinimumConditionalEntropyTransformer, {"frac": 0.01}),
])
def test_infer(simulated_data: np.ndarray, simulated_params: np.ndarray, observed_data: np.ndarray,
               latent_params: np.ndarray, tmp_path: Path, transformer_cls: Type[Transformer],
               transformer_kwargs: Dict[str, Any]) \
        -> None:
    # Set up a dummy configuration.
    config = InferenceConfig(
        transformer_cls,
        0.01,
        transformer_kwargs,
        TestPreprocessor,
    )

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
