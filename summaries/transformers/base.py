from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator
from typing import Any, Optional


class Transformer:
    """
    Type stub for sklearn transformers.
    """
    def fit(self, X: Any, y: Optional[Any] = None) -> Transformer:
        ...  # pragma: no cover

    def transform(self, X: Any) -> Any:
        ...  # pragma: no cover


class PredictorTransformer(BaseEstimator):
    """
    Use a predictor (https://scikit-learn.org/stable/glossary.html#term-predictor) as a trainable,
    supervised transformer.
    """
    def __init__(self, predictor: Transformer, method: str | None = None) -> None:
        self.predictor = predictor
        self.method = method

    def fit(self, data: np.ndarray, params: np.ndarray) -> PredictorTransformer:
        self.predictor.fit(data, params)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        method = self.method or "predict"
        return getattr(self.predictor, method)(data)


class _DataDependentTransformerMixin(BaseEstimator):
    """
    Compressor that depends on the observed data.

    Args:
        observed_data: Vector of raw data or summaries.
    """
    def __init__(self, observed_data: np.ndarray) -> None:
        super().__init__()
        self.observed_data = np.asarray(observed_data)
        if self.observed_data.ndim != 1:
            raise ValueError("Observed data must be a vector; got shape "
                             f"{self.observed_data.shape}.")
        self.n_features_in_, = self.observed_data.shape
