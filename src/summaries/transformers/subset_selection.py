from __future__ import annotations
import logging
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_array, check_X_y
from typing import List, Optional, Tuple

from ..algorithm import NearestNeighborAlgorithm
from ..entropy import estimate_entropy
from .base import _DataDependentTransformerMixin


LOGGER = logging.getLogger(__name__)


class _ExhaustiveSubsetSelectorMixin(BaseEstimator):
    """
    Mixin to perform exhaustive subset selection.
    """
    def __init__(self) -> None:
        super().__init__()
        self.masks_: Optional[np.ndarray] = None
        self.losses_: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray, params: np.ndarray) -> _ExhaustiveSubsetSelectorMixin:
        data, params = check_X_y(data, params, multi_output=True)

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


class _GreedySubsetSelectorMixin(_DataDependentTransformerMixin):
    """
    Mixin to perform greedy subset selection.
    """
    def __init__(self, observed_data: np.ndarray, frac: float | None = None,
                 n_samples: int | None = None, thin: int = 1) -> None:
        super().__init__(observed_data)
        self.frac = frac
        self.n_samples = n_samples
        self.thin = thin
        self.features_: Optional[List[int]] = None
        self.losses_: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray, params: np.ndarray) -> _GreedySubsetSelectorMixin:
        data, params = check_X_y(data, params, multi_output=True)

        # Thin out the data and parameters for more efficient feature selection.
        if self.thin > 1:
            data = data[::self.thin]
            params = params[::self.thin]

        # Iterate over all the features
        candidates = list(range(data.shape[1]))
        self.features_ = []
        kwargs = {
            "frac": self.frac,
            "n_samples": self.n_samples,
            "balanced_tree": False,
            "compact_nodes": False,
        }
        while candidates:
            # Draw samples based on the currently selected features. If there are none, we sample
            # from the prior.
            if self.features_:
                sampler = NearestNeighborAlgorithm(**kwargs)
                sampler.fit(data[:, self.features_], params)
                current, = sampler.predict([self.observed_data[self.features_]])
            else:
                n_samples = self.n_samples or int(self.frac * params.shape[0])
                current = params[np.random.choice(params.shape[0], n_samples, replace=False)]

            # Score all candidates.
            scores = []
            for candidate in candidates:
                features = self.features_ + [candidate]
                sampler = NearestNeighborAlgorithm(**kwargs)
                sampler.fit(data[:, features], params)
                proposed, = sampler.predict([self.observed_data[features]])
                scores.append(self._score_samples(current, proposed))

            # If all scores are bad, we're done here.
            if np.max(scores) == - float("inf"):
                break

            # Get the next best feature.
            candidate = candidates.pop(np.argmax(scores))
            self.features_.append(candidate)

        return self

    def _score_samples(self, current: Optional[np.ndarray], proposed: np.ndarray) -> float:
        """
        Compare two sets of samples.

        Args:
            current: Samples based on the currently selected summaries.
            proposed: Samples based on a proposed additional summary.

        Returns:
            Score indicating the quality of the samples based on the proposed summary; larger is
            better.
        """
        raise NotImplementedError

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.features_ is None:
            raise NotFittedError
        data = check_array(data)
        if not self.features_:
            LOGGER.warning("no features were selected")
            # We return some random data so the algorithm can proceed; this means the ABC posterior
            # will be equal to the prior.
            return np.random.uniform(0, 1, (data.shape[0], 1))
        return data[:, self.features_]


class ApproximateSufficiencyTransformer(_GreedySubsetSelectorMixin):
    """
    Greedy subset selection to find "approximately sufficient" statistics as proposed by Joyce and
    Marjoram (2008).
    """
    def __init__(self, observed_data: np.ndarray, bins: int = 10,
                 range_: Tuple[float, float] | None = None, likelihood_ratio: bool = False,
                 frac: float | None = None, n_samples: int | None = None, alpha: float = 0.05,
                 thin: int = 1) -> None:
        super().__init__(observed_data, frac, n_samples, thin)
        self.bins = bins
        self.likelihood_ratio = likelihood_ratio
        self.range_ = range_
        self.alpha = alpha

    def _score_samples(self, current: np.ndarray | None, proposed: np.ndarray) -> float:
        epsilon = 1e-6
        n_samples, n_params = current.shape
        assert n_params == 1, "only one-dimensional parameters are supported"

        range_ = self.range_
        if self.range_ is None:
            range_ = min(current.min(), proposed.min()), max(current.max(), proposed.max())

        hist_current, _ = np.histogram(current, self.bins, range_)
        hist_proposed, _ = np.histogram(proposed, self.bins, range_)

        if self.likelihood_ratio:
            # Evaluate the likelihood under the assumption that x and y come from the same
            # distribution.
            x = hist_current
            y = hist_proposed
            nx = x.sum(axis=-1, keepdims=True)
            ny = y.sum(axis=-1, keepdims=True)
            p0 = (x + y + epsilon) / (nx + ny)
            ll0 = np.sum((x + y) * np.log(p0), axis=-1)

            # Evaluate the likelihood under the assumption that x and y come from different
            # distributions.
            px = (x + epsilon) / nx
            py = (y + epsilon) / ny
            ll1 = np.sum(x * np.log(px) + y * np.log(py), axis=-1)
            lr = 2 * (ll1 - ll0)

            # Return the likelihood ratio if it is significant so we can prioritize features by how
            # different the posterior looks.
            if stats.chi2(self.bins - 1).cdf(lr) > 1 - self.alpha:
                return lr
        else:
            # Compute z scores. We clip the density below at epsilon to avoid division by zero for
            # empty bins.
            density = np.clip(hist_current / n_samples, epsilon, 1 - epsilon)
            std = np.sqrt(n_samples * density * (1 - density))
            z = (hist_proposed - hist_current) / std

            # Check if any score exceeds the significance level. If yes, we return a random number
            # because the original implementation does not prioritize the features in any way.
            p_value: np.ndarray = stats.norm(0, 1).cdf(z)
            p_value = np.minimum(p_value, 1 - p_value)
            if p_value.min() < self.alpha / 2:
                return np.random.uniform(0, 1)
        return - float("inf")
