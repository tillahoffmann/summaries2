import numpy as np
from scipy.spatial import KDTree
from scipy import special
from sklearn.utils.validation import check_array
from typing import Union


def estimate_entropy(x: np.ndarray, k: int = 4) -> float:
    """
    Estimate the entropy of a point cloud using https://doi.org10.1080/01966324.2003.10737616.

    Args:
        x: Coordinate of points.
        k: Nearest neighbor to use for entropy estimation.

    Returns:
        Estimated entropy of the point cloud.
    """
    # Ensure the point cloud has the right shape.
    x = check_array(x)
    n, p = x.shape

    # Use a KD tree to look up the k^th nearest neighbour.
    tree = KDTree(x)
    distance, _ = tree.query(x, k=k + 1)
    distance = distance[:, -1]

    # Estimate the entropy.
    entropy = (
        p * np.log(np.pi) / 2
        - special.gammaln(p / 2 + 1)
        - special.digamma(k)
        + p * np.log(distance).mean()
    )
    return entropy + np.log(n)


def estimate_divergence(x: np.ndarray, y: np.ndarray, k: int = 4):
    """
    Estimate the Kullback Leibler divergence between two point clouds.
    """
    # Validate input.
    n, p = x.shape
    m, q = y.shape
    assert p == q, "x and y must have the same trailing dimension"

    # Build nearest neighbor trees and query distances.
    xtree = KDTree(x)
    ytree = KDTree(y)

    dxx, _ = xtree.query(x, k=k + 1)
    dxx = dxx[:, -1]
    dxy, _ = ytree.query(x, k=k)
    dxy = dxy[:, -1]
    return p * np.mean(np.log(dxy / dxx)) + np.log(m / (n - 1))


def estimate_mutual_information(
    x: np.ndarray, y: np.ndarray, normalize: Union[bool, str] = False
) -> float:
    """
    Estimate the mutual information between two variables.

    Args:
        x: First variable.
        y: Second variable.
        normalize: Whether to normalize the mutual information. Use `x` to divide by entropy of
            first variable, `y` to divide by entropy of second variable, or `xy` to divide by
            mean entropy of `x` and `y`.
        method: Nearest neighbor method to use for entropy estimation.

    Returns:
        mutual_information: Mutual information estimate (possibly normalised).
    """
    x = check_array(x)
    y = check_array(y)
    entropy_x = estimate_entropy(x)
    entropy_y = estimate_entropy(y)
    entropy_xy = estimate_entropy(np.hstack([x, y]))
    mi = entropy_x + entropy_y - entropy_xy
    if normalize == "x":
        mi /= entropy_x
    elif normalize == "y":
        mi /= entropy_y
    elif normalize == "xy":
        mi /= (entropy_x + entropy_y) / 2
    elif normalize:
        raise NotImplementedError(normalize)
    return mi
