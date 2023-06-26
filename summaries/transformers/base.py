from __future__ import annotations
from functools import partial
import numpy as np
from sklearn.base import BaseEstimator
from typing import Any, Optional, Type, TypeVar


class Transformer:
    """
    Type stub for sklearn transformers.
    """
    def fit(self, X: Any, y: Optional[Any] = None) -> Transformer:
        ...  # pragma: no cover

    def transform(self, X: Any) -> Any:
        ...  # pragma: no cover


T = TypeVar("T")


def as_transformer(cls: T, *, _method: str | None = None) -> T:
    """
    Use a predictor (https://scikit-learn.org/stable/glossary.html#term-predictor) as a trainable,
    supervised transformer.

    Args:
        cls: Type of the predictor.
        _method: Predictor method to use for transforming data (determined automatically by
            default).

    Returns:
        Predictor type whose prediction method acts as a transformer.
    """
    return partial(_PredictorTransformer, cls, _method=_method)


class _PredictorTransformer(BaseEstimator):
    """
    Use a predictor (https://scikit-learn.org/stable/glossary.html#term-predictor) as a trainable,
    supervised transformer.
    """
    def __init__(self, cls: Type[BaseEstimator], *, _method: str | None = None, **kwargs: Any) \
            -> None:
        self.cls = cls
        self.kwargs = kwargs
        self.predictor = self.cls(**self.kwargs)
        self._method = _method

    def fit(self, data: np.ndarray, params: np.ndarray) -> _PredictorTransformer:
        self.predictor.fit(data, params)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        method = self._method or "predict"
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
