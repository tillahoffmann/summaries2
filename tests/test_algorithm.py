import numpy as np
import pytest
from scipy import stats
from sklearn.exceptions import NotFittedError
from summaries.algorithm import NearestNeighborAlgorithm


def test_posterior_mean_correlation(simulated_data: np.ndarray, simulated_params: np.ndarray,
                                    observed_data: np.ndarray, latent_params: np.ndarray) -> None:
    sampler = NearestNeighborAlgorithm(0.001).fit(simulated_data, simulated_params)
    posterior_mean = sampler.predict(observed_data).mean(axis=1)
    pearsonr = stats.pearsonr(posterior_mean.ravel(), latent_params.ravel())
    assert pearsonr.statistic > 0.8 and pearsonr.pvalue < 0.01


def test_nearest_neighbor_not_fitted() -> None:
    with pytest.raises(NotFittedError):
        NearestNeighborAlgorithm(0.01).predict(None)
