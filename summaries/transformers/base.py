from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from typing import Optional, Type, TypeVar


T = TypeVar("T")


def as_transformer(cls: Type[T], *, method: Optional[str] = None) -> Type[T]:
    """
    Use a predictor (https://scikit-learn.org/stable/glossary.html#term-predictor) as a trainable,
    supervised transformer.
    """
    for candidate in ["predict", "predict_proba"]:
        method = method or candidate

    class _Transformer(cls):
        def transform(self, data: np.ndarray) -> np.ndarray:
            transformed = getattr(self, method)(data)
            return check_array(transformed)

    _Transformer.__name__ = f"Transformer{cls.__name__}"
    return _Transformer


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
