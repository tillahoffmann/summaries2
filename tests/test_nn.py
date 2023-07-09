import pytest
from summaries.nn import MeanPool, NegLogProbLoss, SequentialWithKeywords
import torch
from torch import eye
from torch.distributions import Wishart
import torch_geometric.nn


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_neg_log_prob_loss(reduction: str) -> None:
    loss = NegLogProbLoss(reduction)
    dist = Wishart(999, eye(3))
    x = dist.sample([13])
    value = loss(dist, x)
    assert value.shape == ((13,) if reduction == "none" else ())


def test_mean_pool() -> None:
    mean_pool = MeanPool()
    assert mean_pool(torch.randn(6, 5, 7)).shape == (6, 7)


def test_sequential_with_keywords() -> None:
    edge_index = torch.randint(0, 100, (2, 10))
    x = torch.randn(100, 2)

    conv = torch_geometric.nn.GINConv(torch.nn.LazyLinear(5))
    tanh = torch.nn.Tanh()
    y = tanh(conv(x, edge_index=edge_index))

    z = SequentialWithKeywords(conv, tanh)(x, edge_index=edge_index)

    torch.testing.assert_close(y, z)
