import numpy as np
import pytest


def sample_params(n: int, p: int) -> np.ndarray:
    return np.random.normal(0, 1, (n, p))


def sample_data(params: np.ndarray) -> np.ndarray:
    n = params.shape[0]
    x = params + np.random.normal(0, .1, (n, 2))
    return np.hstack([x, np.random.normal(0, 10, (n, 1))])


@pytest.fixture(params=[1, 2])
def n_params(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def simulated_params(n_params: int) -> np.ndarray:
    return sample_params(100_000, n_params)


@pytest.fixture
def simulated_data(simulated_params: np.ndarray) -> np.ndarray:
    return sample_data(simulated_params)


@pytest.fixture
def latent_params(n_params: int) -> np.ndarray:
    return sample_params(100, n_params)


@pytest.fixture
def observed_data(latent_params: np.ndarray) -> np.ndarray:
    return sample_data(latent_params)
