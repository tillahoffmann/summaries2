import inspect
from torch import nn, Tensor
from torch.distributions import Distribution
from typing import Any, Literal


class NegLogProbLoss(nn.Module):
    """
    Measure the negative log probability loss.
    """
    def __init__(self, reduction: Literal["none", "mean", "sum"] = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, distribution: Distribution, params: Tensor) -> Tensor:
        loss = - distribution.log_prob(params)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError(f"{self.reduction} is not a valid reduction")


class MeanPool(nn.Module):
    """
    Mean-pool features.
    """
    def __init__(self, axis: int = -2) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(axis=self.axis)


class SequentialWithKeywords(nn.Module):
    """
    Apply modules sequentially with optional keyword arguments.
    """
    def __init__(self, *layers: nn.Module) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Any, **kwargs: Any) -> Any:
        for layer in self.layers:
            signature = inspect.Signature.from_callable(layer.forward)
            keys = set(kwargs) & set(signature.parameters)
            x = layer(x, **{key: kwargs[key] for key in keys})
        return x
