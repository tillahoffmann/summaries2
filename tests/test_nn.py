import pytest
from summaries.nn import NegLogProbLoss
from torch import eye
from torch.distributions import Wishart


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_neg_log_prob_loss(reduction: str) -> None:
    loss = NegLogProbLoss(reduction)
    dist = Wishart(999, eye(3))
    x = dist.sample([13])
    value = loss(dist, x)
    assert value.shape == ((13,) if reduction == "none" else ())
