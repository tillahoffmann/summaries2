import numpy as np
import torch
from torch import distributions, nn

from ..nn import MeanPool
from ..transformers import NeuralTransformer


def simulate_benchmark(params: np.ndarray, n_observations: int,
                       random_state: np.random.RandomState | None = None,
                       n_noise_features: int = 1) -> np.ndarray:
    """
    Simulate benchmark data.

    Args:
        params: Univariate parameter to infer.
        n_observations: Number of independent observations.
        random_state: Random number generator state.
        n_noise_features: Number of white noise features per independent observation.
    """
    random_state = random_state or np.random
    u = np.tanh(params)
    data = random_state.normal(u, np.sqrt(1 - u ** 2), (*params.shape[:-1], n_observations))
    data *= 2 * random_state.binomial(1, 0.5, data.shape) - 1
    return np.concatenate([
        data[..., None],
        random_state.normal(0, 1, data.shape + (n_noise_features,))
    ], axis=-1)


class _BenchmarkTransformer(NeuralTransformer):
    """
    Learnable transformer for the coalescent problem without the "head" of the network.
    """
    def __init__(self) -> None:
        transformer = nn.Sequential(
            nn.LazyLinear(16),
            nn.Tanh(),
            nn.LazyLinear(16),
            nn.Tanh(),
            nn.LazyLinear(1),
            MeanPool(),
            nn.Tanh(),
        )
        super().__init__(transformer)


class BenchmarkPosteriorMeanTransformer(_BenchmarkTransformer):
    """
    Learnable transformer with posterior mean predictive "head".
    """
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Sequential(
            self.transformer,
            nn.Tanh(),
            nn.LazyLinear(16),
            nn.Tanh(),
            nn.LazyLinear(1),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.transformer(data)


class BenchmarkPosteriorMixtureDensityTransformer(_BenchmarkTransformer):
    """
    Learnable transformer with conditional posterior density estimation "head" based on mixture
    density networks.
    """
    def __init__(self, n_components: int = 10) -> None:
        super().__init__()
        self.n_components = n_components
        self.mixture_parameters = nn.ModuleDict({
            key: nn.Sequential(
                nn.LazyLinear(16),
                nn.Tanh(),
                nn.LazyLinear(n_components),
            ) for key in ["logits", "locs", "scales"]
        })

    def forward(self, data: torch.Tensor, transformed: torch.Tensor | None = None) -> torch.Tensor:
        transformed = self.transformer(data) if transformed is None else transformed

        logits: torch.Tensor = self.mixture_parameters["logits"](transformed)
        mixture_dist = distributions.Categorical(logits=logits)

        locs: torch.Tensor = self.mixture_parameters["locs"](transformed)[..., None]
        scales: torch.Tensor = self.mixture_parameters["scales"](transformed).exp()[..., None]
        component_dist = distributions.Independent(distributions.Normal(locs, scales), 1)
        mixture = distributions.MixtureSameFamily(mixture_dist, component_dist)
        return mixture
