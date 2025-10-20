from __future__ import annotations
from sklearn.base import BaseEstimator
from snippets.tensor_data_loader import TensorDataLoader
import torch
from torch.nn import Module
from torch.utils.data import TensorDataset
from typing import Any


class NeuralTransformer(Module, BaseEstimator):
    """
    Transform data using a neural network.

    Args:
        transformer: Transformer network.
        data_as_tensor: Cast data to tensors.
        batch_size: Size of batches to be used for transforming data.
    """

    def __init__(
        self,
        transformer: Module,
        data_as_tensor: bool = True,
        batch_size: int | None = None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.data_as_tensor = data_as_tensor
        self.batch_size = batch_size

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def fit(self, data: torch.Tensor, params: torch.Tensor) -> NeuralTransformer:
        return self

    def transform(self, data: Any) -> torch.Tensor:
        if self.data_as_tensor:
            data = torch.as_tensor(data, dtype=torch.get_default_dtype())

        # TODO: this should just be self.batch_size but some of the older trained models don't have
        # said attribute, and retraining would take a while.
        batch_size = getattr(self, "batch_size", None)
        if batch_size is None:
            return self.transformer(data)

        # We need tensor input for batched transforms.
        if not torch.is_tensor(data):
            raise TypeError(
                f"data must be a tensor for batched transformations; got {type(data)}"
            )
        loader = TensorDataLoader(
            TensorDataset(data), batch_size=batch_size, shuffle=False
        )
        return torch.concatenate([self.transformer(*batch) for batch in loader], axis=0)
