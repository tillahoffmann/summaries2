from __future__ import annotations
from typing import Any, Optional

from .base import as_transformer
from .exhaustive_subset_selection import MinimumConditionalEntropyTransformer
from .nn import NeuralTransformer


__all__ = [
    "as_transformer",
    "MinimumConditionalEntropyTransformer",
    "NeuralTransformer",
]


class Transformer:
    def fit(self, X: Any, y: Optional[Any] = None) -> Transformer:
        ...  # pragma: no cover

    def transform(self, X: Any) -> Any:
        ...  # pragma: no cover
