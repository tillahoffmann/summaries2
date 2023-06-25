from torch import nn, Tensor
from torch.distributions import AffineTransform, Beta, Categorical, Independent, \
    MixtureSameFamily, TransformedDistribution

from ..transformers import NeuralTransformer


class _CoalTransformer(NeuralTransformer):
    """
    Learnable transformer for the coalescent problem without the "head" of the network.
    """
    def __init__(self) -> None:
        transformer = nn.Sequential(
            nn.LazyLinear(16),
            nn.Tanh(),
            nn.LazyLinear(16),
            nn.Tanh(),
            nn.LazyLinear(2),
            nn.Tanh(),
        )
        super().__init__(transformer)


class CoalPosteriorMeanTransformer(_CoalTransformer):
    """
    Learnable transformer with posterior mean predictive "head".
    """
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Sequential(
            self.transformer,
            nn.LazyLinear(16),
            nn.Tanh(),
            nn.LazyLinear(2),
        )

    def forward(self, data: Tensor) -> Tensor:
        return self.transformer(data)


class CoalPosteriorMixtureDensityTransformer(_CoalTransformer):
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
                nn.LazyLinear(size * n_components),
            ) for (size, key) in [(1, "logits"), (2, "concentration1s"), (2, "concentration0s")]
        })

    def forward(self, data: Tensor) -> Tensor:
        transformed = self.transformer(data)

        logits: Tensor = self.mixture_parameters["logits"](transformed)
        mixture_dist = Categorical(logits=logits)

        concentration1s: Tensor = self.mixture_parameters["concentration1s"](transformed)
        concentration1s = concentration1s.reshape((-1, self.n_components, 2)).exp()
        concentration0s: Tensor = self.mixture_parameters["concentration0s"](transformed)
        concentration0s = concentration0s.reshape((-1, self.n_components, 2)).exp()
        component_dist = Beta(concentration1s, concentration0s)

        # Rescale the distribution to the [0, 10] domain used by the simulations and reinterpret the
        # last batch dimension as an event dimension.
        component_dist = TransformedDistribution(component_dist, AffineTransform(0, 10))
        component_dist = Independent(component_dist, 1)

        return MixtureSameFamily(mixture_dist, component_dist)
