import numpy as np
import pytest
from scipy import stats
from sklearn.exceptions import NotFittedError
from summaries.algorithm import NearestNeighborAlgorithm


def test_posterior_mean_correlation(simulated_data: np.ndarray, simulated_params: np.ndarray,
                                    observed_data: np.ndarray, latent_params: np.ndarray) -> None:
    sampler = NearestNeighborAlgorithm(frac=0.001).fit(simulated_data, simulated_params)
    posterior_mean = sampler.predict(observed_data).mean(axis=1)
    pearsonr = stats.pearsonr(posterior_mean.ravel(), latent_params.ravel())
    assert pearsonr.statistic > 0.8 and pearsonr.pvalue < 0.01


def test_nearest_neighbor_not_fitted() -> None:
    with pytest.raises(NotFittedError):
        NearestNeighborAlgorithm(frac=0.01).predict(None)


@pytest.mark.parametrize("n_samples", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 7])
def test_nearest_neighbor_single_sample(n_samples: int, batch_size: int) -> None:
    sampler = NearestNeighborAlgorithm(n_samples=n_samples)
    sampler.fit(np.random.normal(0, 1, (100, 3)), np.random.normal(0, 1, (100, 2)))
    samples = sampler.predict(np.random.normal(0, 1, (batch_size, 3)))
    assert samples.shape == (batch_size, n_samples, 2)


def test_mutually_exclusive_kwargs() -> None:
    with pytest.raises(ValueError, match="Exactly one of"):
        NearestNeighborAlgorithm()
    with pytest.raises(ValueError, match="Exactly one of"):
        NearestNeighborAlgorithm(frac=0.3, n_samples=3)
