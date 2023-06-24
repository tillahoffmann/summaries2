import numpy as np
import pytest
from scipy import stats
from summaries.entropy import estimate_divergence, estimate_entropy, estimate_mutual_information


N_SAMPLES = 10000


def test_estimate_entropy() -> None:
    scale = 7
    dist = stats.norm(0, scale)
    x = dist.rvs((N_SAMPLES, 1))
    actual = estimate_entropy(x)
    expected = dist.entropy()
    assert abs(actual - expected) < .1


@pytest.mark.parametrize('normalize', ['x', 'y', 'xy', False])
def test_estimate_mutual_information(normalize) -> None:
    cov = np.asarray([[1, .7], [.7, 2]])
    x = np.random.multivariate_normal(np.zeros(2), cov, size=(1, N_SAMPLES))
    actual = estimate_mutual_information(*x.T, normalize=normalize)
    expected = (np.log(np.diag(cov)).sum() - np.linalg.slogdet(cov)[1]) / 2
    if normalize == 'x':
        norm = stats.norm(0, cov[0, 0]).entropy()
    elif normalize == 'y':
        norm = stats.norm(0, cov[1, 1]).entropy()
    elif normalize == 'xy':
        norm = stats.norm(0, np.diag(cov)).entropy().mean()
    elif not normalize:
        norm = 1
    else:
        raise NotImplementedError(normalize)
    expected /= norm
    assert abs(actual - expected) < .1


@pytest.mark.parametrize('p', [1, 2])
def test_estimate_divergence(p):
    n = 100000
    m = 100000
    loc1 = 0
    loc2 = 1.4
    scale1 = 0.7
    scale2 = 1.3
    var1 = scale1 ** 2
    var2 = scale2 ** 2
    expected = p * ((loc1 - loc2) ** 2 / var2 + var1 / var2 - 1 - np.log(var1 / var2)) / 2
    x = stats.norm(loc1, scale1).rvs((n, p))
    y = stats.norm(loc2, scale2).rvs((m, p))
    actual = estimate_divergence(x, y, k=7)
    assert abs(expected - actual) < 0.05
