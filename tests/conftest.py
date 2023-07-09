import numpy as np
import pytest
import subprocess
from textwrap import dedent


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


class shared:
    """
    Shared code to be used in tests (cf. https://stackoverflow.com/a/76082875/1150961).
    """
    def check_pickle_loadable(path: str) -> None:
        """
        Try to load a pickled file in a separate process to verify its un-pickle-ability.
        """
        code = f"""
        import pickle
        with open("{path}", "rb") as fp:
            pickle.load(fp)
        """
        process = subprocess.Popen(["python", "-"], stdin=subprocess.PIPE, text=True,
                                   stderr=subprocess.PIPE)
        _, stderr = process.communicate(dedent(code))
        if process.returncode:
            raise RuntimeError(f"'{path}' cannot be unpickled: \n{stderr}")


pytest.shared = shared
