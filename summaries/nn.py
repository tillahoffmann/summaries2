from torch import nn, Tensor
from torch.distributions import Distribution
from typing import Literal


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
