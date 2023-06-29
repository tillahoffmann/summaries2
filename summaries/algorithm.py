from __future__ import annotations
import numpy as np
from scipy.spatial import KDTree
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_array
from typing import Any, Optional


class NearestNeighborAlgorithm(BaseEstimator):
    """
    Draw approximate posterior samples using a nearest neighbor algorithm.

    Args:
        frac: Fraction of samples to return as approximate posterior samples (mutually exclusive
            with `n_samples).
        n_samples: Number of samples to draw (mutually exclusive with `frac`).
        minkowski_norm: Minkowski p-norm to use for queries (defaults to Euclidean distances).
        **kdtree_kwargs: Keyword arguments passed to the KDTree constructor.
    """
    def __init__(self, *, frac: float | None = None, n_samples: int | None = None,
                 minkowski_norm: float = 2, **kdtree_kwargs) -> None:
        super().__init__()
        if (frac is None) == (n_samples is None):
            raise ValueError("Exactly one of `frac` and `n_samples` must be given.")
        self.frac = frac
        self.n_samples = n_samples
        self.minkowski_norm = minkowski_norm
        self.kdtree_kwargs = kdtree_kwargs

        self.tree_: Optional[KDTree] = None
        self.params_: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray, params: np.ndarray) -> NearestNeighborAlgorithm:
        """
        Construct a :class:`.KDTree` for fast nearest neighbor search for sampling parameters.

        Args:
            data: Simulated data or summary statistics used to build the tree.
            params: Dictionary of parameters used to generate the corresponding `data` realization.
        """
        self.tree_ = KDTree(data, **self.kdtree_kwargs)
        self.params_ = params
        return self

    def predict(self, data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Draw approximate posterior samples.

        Args:
            data: Data to condition on with shape `(batch_size, n_features)`.
            **kwargs: Keyword arguments passed to the KDTree query method.

        Returns:
            Dictionary of posterior samples. Each value has shape
            `(batch_size, n_samples, *event_shape)`, where `event_shape` is the basic shape of the
            corresponding parameter.
        """
        # Validate the state and input arguments.
        if self.tree_ is None:
            raise NotFittedError

        data = check_array(data)
        n_samples = self.n_samples or int(self.frac * self.tree_.n)
        _, idx = self.tree_.query(data, k=n_samples, p=self.minkowski_norm, **kwargs)
        # Explicitly reshape because `query` drops on dimension if the number of samples is one.
        idx = idx.reshape((*data.shape[:-1], n_samples))

        return self.params_[idx]
