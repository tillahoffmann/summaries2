import numpy as np
import pytest
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from summaries.algorithm import NearestNeighborAlgorithm
from summaries.transformers import as_transformer
from typing import Any, Dict, Type


@pytest.mark.parametrize(
    "predictor_cls, predictor_kwargs",
    [
        (LinearRegression, {}),
        (MLPRegressor, {"max_iter": 1}),
    ],
)
def test_pipeline_posterior_mean_correlation(
    simulated_data: np.ndarray,
    simulated_params: np.ndarray,
    observed_data: np.ndarray,
    latent_params: np.ndarray,
    predictor_cls: Type[BaseEstimator],
    predictor_kwargs: Dict[str, Any],
) -> None:
    pipeline = Pipeline(
        [
            ("standardize_data", StandardScaler()),
            ("learn_posterior_mean", as_transformer(predictor_cls)(**predictor_kwargs)),
            ("standardize_summaries", StandardScaler()),
            ("sample_posterior", NearestNeighborAlgorithm(frac=0.01)),
        ]
    )
    pipeline.fit(simulated_data, simulated_params)
    posterior_mean = pipeline.predict(observed_data).mean(axis=1)

    pearsonr = stats.pearsonr(posterior_mean.ravel(), latent_params.ravel())
    assert pearsonr.statistic > 0.8 and pearsonr.pvalue < 0.01
