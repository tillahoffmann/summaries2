import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from summaries.transformers import ApproximateSufficiencyTransformer, as_transformer, \
    MinimumConditionalEntropyTransformer, NeuralTransformer
from summaries.transformers.base import _DataDependentTransformerMixin, Transformer
import torch
from torch.nn import Identity
from typing import Dict, Type


@pytest.mark.parametrize("transformer_cls, kwargs", [
    (as_transformer(LinearRegression), {"fit_intercept": False}),
    (MinimumConditionalEntropyTransformer, {"frac": 0.01}),
    (ApproximateSufficiencyTransformer, {"frac": 0.01}),
    (ApproximateSufficiencyTransformer, {"frac": 0.01, "range_": (-1, 1)}),
    (ApproximateSufficiencyTransformer, {"frac": 0.01, "likelihood_ratio": True}),
    (NeuralTransformer, {"transformer": Identity()}),
    # This isn't implemented here, but we test it to see if it fits into the framework.
    (PLSRegression, {}),
])
def test_transformer(transformer_cls: Type[Transformer], kwargs: Dict, simulated_data: np.ndarray,
                     simulated_params: np.ndarray, observed_data: np.ndarray) -> None:

    # Create the transformer and verify it does not transform without fitting (except pretrained
    # neural transformers).
    if isinstance(transformer_cls, Type) \
            and issubclass(transformer_cls, _DataDependentTransformerMixin):
        kwargs["observed_data"] = observed_data[0]
    transformer = transformer_cls(**kwargs)

    # Skip multidimensional data for approximate sufficiency.
    if isinstance(transformer, ApproximateSufficiencyTransformer) and simulated_params.shape[1] > 1:
        pytest.skip("approximate sufficiency only supports one-dimensional parameters")

    if not isinstance(transformer, NeuralTransformer):
        with pytest.raises(NotFittedError):
            transformer.transform(simulated_data)

    # Fit, apply, and verify shapes.
    transformer.fit(simulated_data, simulated_params)
    transformed = transformer.transform(simulated_data)
    assert transformed.shape[0] == simulated_data.shape[0]
    assert transformed.shape[1] <= simulated_data.shape[1]

    if isinstance(transformer, LinearRegression):
        # Verify the learned coefficients.
        coef = np.squeeze(transformer.coef_)
        if simulated_params.shape[1] == 1:
            np.testing.assert_allclose(coef[:2], 0.5, atol=0.05)
            np.testing.assert_allclose(coef[2], 0, atol=0.05)
        else:
            np.testing.assert_allclose(coef[0, 0], 1, atol=0.05)
            np.testing.assert_allclose(coef[0, 1:], 0, atol=0.05)
            np.testing.assert_allclose(coef[1, 1], 1, atol=0.05)
            np.testing.assert_allclose(coef[1, [0, 2]], 0, atol=0.05)
    elif isinstance(transformer, MinimumConditionalEntropyTransformer):
        # Verify the trailing noise feature is ignored.
        np.testing.assert_array_equal(transformer.best_mask_, [True, True, False])


def test_data_dependent_transformer_invalid_shape() -> None:
    with pytest.raises(ValueError, match="must be a vector"):
        MinimumConditionalEntropyTransformer(np.zeros((3, 2)), frac=0.1)


def test_neural_transformer_batch() -> None:
    class _Identity(Identity):
        def __init__(self):
            super().__init__()
            self.n_calls = 0

        def forward(self, *args, **kwargs):
            self.n_calls += 1
            return super().forward(*args, **kwargs)

    identity = _Identity()
    transformer = NeuralTransformer(identity, batch_size=10, data_as_tensor=False)
    transformer.transform(torch.randn(27, 3))
    assert identity.n_calls == 3

    with pytest.raises(TypeError, match="must be a tensor"):
        transformer.transform(None)
