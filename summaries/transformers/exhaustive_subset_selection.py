from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_array
from typing import Optional

from ..algorithm import NearestNeighborAlgorithm
from ..entropy import estimate_entropy
from .base import _DataDependentTransformerMixin


class _ExhaustiveSubsetSelectorMixin(BaseEstimator):
    """
    Mixin to perform exhaustive subset selection.
    """
    def __init__(self) -> None:
        super().__init__()
        self.masks_: Optional[np.ndarray] = None
        self.losses_: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray, params: np.ndarray) -> _ExhaustiveSubsetSelectorMixin:
        data = check_array(data)
        params = check_array(params)

        # Construct the binary mask.
        n_masks = 2 ** self.n_features_in_
        self.masks_ = (np.arange(1, n_masks)[:, None] >> np.arange(self.n_features_in_)) & 1 > 0

        # Iterate over all masks and record the associated loss.
        self.losses_ = np.asarray([self._evaluate_mask(data, params, mask) for mask in self.masks_])
        return self

    def _evaluate_mask(self, data: np.ndarray, params: np.ndarray, mask: np.ndarray) -> float:
        """
        Evaluate a mask returning a lower value for better masks.
        """
        raise NotImplementedError

    @property
    def best_mask_(self):
        """
        Mask with the smallest loss value.
        """
        if self.losses_ is None:
            raise NotFittedError
        return self.masks_[np.argmin(self.losses_)]

    def transform(self, data: np.ndarray) -> np.ndarray:
        return check_array(data)[:, self.best_mask_]


class MinimumConditionalEntropyTransformer(_DataDependentTransformerMixin,
                                           _ExhaustiveSubsetSelectorMixin):
    """
    Exhaustive subset selection to minimize the conditional (given the observed data) posterior
    entropy as proposed by Nunes and Balding (2010).

    Args:
        observed_data: Vector of raw data or summaries.
        frac: Passed to the internal nearest neighbor sampler for estimating the entropy.
        n_samples: Passed to the internal nearest neighbor sampler for estimating the entropy.
        thin: Use only every `thin` samples to draw posterior samples for estimating the entropy.
    """
    def __init__(self, observed_data: np.ndarray, *, frac: float | None = None,
                 n_samples: int | None = None, thin: int = 1) -> None:
        super().__init__(observed_data)
        self.frac = frac
        self.n_samples = n_samples
        self.thin = thin

    def _evaluate_mask(self, data, params: np.ndarray, mask: np.ndarray) -> float:
        # Building the tree is relatively expensive compared with querying because we only query
        # once. The keyword arguments try to reduce the tree build time.
        sampler = NearestNeighborAlgorithm(frac=self.frac, n_samples=self.n_samples,
                                           balanced_tree=False, compact_nodes=False)
        sampler.fit(data[::self.thin, mask], params[::self.thin])
        # Add a batch dimension to the observed data, draw samples, and remove the batch dimension.
        samples, = sampler.predict([self.observed_data[mask]])
        return estimate_entropy(samples)
