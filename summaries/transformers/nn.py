from __future__ import annotations
from sklearn.base import BaseEstimator
from torch import as_tensor, get_default_dtype, Tensor
from torch.nn import Module
from typing import Any


class NeuralTransformer(Module, BaseEstimator):
    """
    Transform data using a neural network.

    Args:
        transformer: Transformer network.
        data_as_tensor: Cast data to tensors.
    """
    def __init__(self, transformer: Module, data_as_tensor: bool = True) -> None:
        super().__init__()
        self.transformer = transformer
        self.data_as_tensor = data_as_tensor

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def fit(self, data: Tensor, params: Tensor) -> NeuralTransformer:
        return self

    def transform(self, data: Any) -> Tensor:
        if self.data_as_tensor:
            data = as_tensor(data, dtype=get_default_dtype())
        return self.transformer(data)
