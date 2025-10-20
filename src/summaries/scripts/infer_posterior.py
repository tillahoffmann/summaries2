from argparse import ArgumentParser
from datetime import datetime
import json
import numpy as np
from pathlib import Path
import pickle
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from snippets.timer import Timer
from torch import no_grad
import torch_geometric.data
import torch_geometric.utils
from tqdm import tqdm
from typing import Any, Dict, List

from ..algorithm import NearestNeighborAlgorithm
from ..experiments.tree import predecessors_to_datasets
from ..transformers import (
    ApproximateSufficiencyTransformer,
    as_transformer,
    MinimumConditionalEntropyTransformer,
    NeuralTransformer,
    Transformer,
)
from .base import resolve_path


class Args:
    config: str
    simulated: Path
    observed: Path
    output: Path
    transformer_kwargs: Dict[str, Any] | None
    n_samples: int | None


class InferenceConfig:
    """
    Base class for inference configurations.
    """

    N_SAMPLES: float | None = None
    IS_DATA_DEPENDENT: bool = False

    def __init__(self, args: Args) -> None:
        self.args = args

    def create_transformer(self, observed_data: Any | None = None) -> Transformer:
        raise NotImplementedError

    def create_preprocessor(self) -> Transformer | None:
        return None

    @property
    def n_samples(self):
        return self.args.n_samples or self.N_SAMPLES


class PriorConfig(InferenceConfig):
    N_SAMPLES = 1_000

    def create_transformer(self, observed_data: Any | None = None) -> Transformer:
        return FunctionTransformer(self._evaluate_random_features)

    def _evaluate_random_features(self, X: np.ndarray) -> np.ndarray:
        # Whatever randomness we have here really doesn't matter because we use these features to
        # pick anything from the reference table.
        return np.random.normal(0, 1, (X.shape[0], 1))


class CoalescentConfig(InferenceConfig):
    N_SAMPLES = 1_000


class CoalescentMinimumConditionalEntropyConfig(CoalescentConfig):
    IS_DATA_DEPENDENT = True

    def create_transformer(self, observed_data: np.ndarray) -> Transformer:
        return MinimumConditionalEntropyTransformer(
            observed_data, n_samples=self.n_samples, thin=10
        )


class CoalescentLinearPosteriorMeanConfig(CoalescentConfig):
    def create_transformer(self, observed_data: Any | None = None) -> Transformer:
        return as_transformer(LinearRegression)()


class CoalescentNeuralConfig(CoalescentConfig):
    def create_transformer(self, observed_data: Any | None = None) -> Transformer:
        with open(self.args.transformer_kwargs["transformer"], "rb") as fp:
            return pickle.load(fp)["transformer"]


class CoalescentExpertSummaryConfig(CoalescentConfig):
    def create_transformer(self, observed_data: Any | None = None) -> Transformer:
        return FunctionTransformer()


class CoalescentPLSConfig(CoalescentConfig):
    def create_transformer(self, observed_data: Any | None = None) -> Transformer:
        # Seven expert summaries.
        return GridSearchCV(PLSRegression(), {"n_components": np.arange(1, 8)})


class BenchmarkConfig(InferenceConfig):
    N_SAMPLES = 1_000

    def create_preprocessor(self) -> Transformer:
        return Pipeline(
            [
                ("candidate_summaries", FunctionTransformer(self._evaluate_summaries)),
                ("standardize", StandardScaler()),
            ]
        )

    def _evaluate_summaries(self, observed_data: np.ndarray) -> np.ndarray:
        # The dataset has shape (n_examples, n_observations, 1 + n_noise_features), and we evaluate
        # the first few moments to get (n_examples, (1 + n_noise_features) * n_moments) as candidate
        # summaries.
        assert observed_data.ndim == 3
        n_examples, n_observations, n_features = observed_data.shape
        n_moments = 3
        expanded = observed_data[..., None] ** (2 * (1 + np.arange(n_moments)))
        assert expanded.shape == (n_examples, n_observations, n_features, n_moments)
        return expanded.mean(axis=1).reshape((n_examples, n_features * n_moments))


class BenchmarkExpertSummaryConfig(BenchmarkConfig):
    def create_transformer(self, observed_data: Any | None = None) -> Transformer:
        return FunctionTransformer()


class BenchmarkMinimumConditionalEntropyConfig(BenchmarkConfig):
    IS_DATA_DEPENDENT = True

    def create_transformer(self, observed_data: np.ndarray) -> Transformer:
        return MinimumConditionalEntropyTransformer(
            observed_data, n_samples=self.n_samples, thin=10
        )


class BenchmarkApproximateSufficiencyConfig(BenchmarkConfig):
    IS_DATA_DEPENDENT = True

    def create_transformer(self, observed_data: Any | None = None) -> Transformer:
        return ApproximateSufficiencyTransformer(
            observed_data, n_samples=self.n_samples, range_=(-3, 3), thin=10
        )


class BenchmarkApproximateSufficiencyLikelihoodRatioConfig(BenchmarkConfig):
    IS_DATA_DEPENDENT = True

    def create_transformer(self, observed_data: Any | None = None) -> Transformer:
        return ApproximateSufficiencyTransformer(
            observed_data,
            n_samples=self.n_samples,
            thin=10,
            likelihood_ratio=True,
            range_=(-3, 3),
        )


class BenchmarkLinearPosteriorMeanConfig(BenchmarkConfig):
    def create_transformer(self, observed_data: Any | None = None) -> Transformer:
        return as_transformer(LinearRegression)()


class BenchmarkPLSConfig(BenchmarkExpertSummaryConfig):
    def create_transformer(self, observed_data: Any | None = None) -> Transformer:
        # First three moments times two observations (one signal, one noise) for a total of six
        # features.
        return GridSearchCV(PLSRegression(), {"n_components": np.arange(1, 7)})


class BenchmarkNeuralConfig(BenchmarkConfig):
    BATCH_SIZE = 1_000

    def create_transformer(self, observed_data: Any | None = None) -> Transformer:
        with open(self.args.transformer_kwargs["transformer"], "rb") as fp:
            transformer: NeuralTransformer = pickle.load(fp)["transformer"]
        # Patch the batch size to avoid memory issues when we process the reference table. For the
        # "large" benchmark dataset we have up to 10^6 (n_examples) * 10^2 (n_observations)
        # * 10^{1 to 2} (hidden layer representations). That ends up being somewhere between 4 and
        # 40 Gb of memory at float32 precision.
        transformer.batch_size = self.BATCH_SIZE
        return transformer

    def create_preprocessor(self) -> None:
        # Overwrite the preprocessor to ensure we pass the raw data to the neural networks.
        return None


class TreeKernelConfig(InferenceConfig):
    N_SAMPLES = 1000


class TreeKernelExpertSummaryConfig(TreeKernelConfig):
    """
    Draw samples using "expert" summary statistics designed for growing trees after standardizing.
    """

    def create_preprocessor(self) -> Transformer:
        return StandardScaler()

    def create_transformer(self, observed_data: Any | None = None) -> Transformer:
        return FunctionTransformer()


class TreeKernelLinearPosteriorMeanConfig(TreeKernelExpertSummaryConfig):
    def create_transformer(self, observed_data: Any | None = None) -> Transformer:
        return as_transformer(LinearRegression)()


class TreeKernelPLSConfig(TreeKernelExpertSummaryConfig):
    def create_transformer(self, observed_data: Any | None = None) -> Transformer:
        return GridSearchCV(PLSRegression(), {"n_components": np.arange(1, 6)})


class TreeKernelMinimumConditionalEntropyConfig(TreeKernelExpertSummaryConfig):
    IS_DATA_DEPENDENT = True

    def create_transformer(self, observed_data: Any | None = None) -> Transformer:
        return MinimumConditionalEntropyTransformer(
            observed_data, n_samples=self.n_samples
        )


class TreeKernelApproximateSufficiencyConfig(TreeKernelExpertSummaryConfig):
    IS_DATA_DEPENDENT = True

    def create_transformer(self, observed_data: Any | None = None) -> Transformer:
        return ApproximateSufficiencyTransformer(
            observed_data, range_=(0, 2), n_samples=self.n_samples
        )


class TreeKernelApproximateSufficiencyLikelihoodRatioConfig(
    TreeKernelExpertSummaryConfig
):
    IS_DATA_DEPENDENT = True

    def create_transformer(self, observed_data: Any | None = None) -> Transformer:
        return ApproximateSufficiencyTransformer(
            observed_data,
            range_=(0, 2),
            n_samples=self.n_samples,
            likelihood_ratio=True,
        )


class TreeKernelNeuralConfig(TreeKernelConfig):
    def create_transformer(self, observed_data: Any | None = None) -> Transformer:
        with open(self.args.transformer_kwargs["transformer"], "rb") as fp:
            return pickle.load(fp)["transformer"]

    def create_preprocessor(self) -> Transformer | None:
        return FunctionTransformer(self._predecessors_to_batch)

    def _predecessors_to_batch(self, data: np.ndarray) -> torch_geometric.data.Data:
        datasets = predecessors_to_datasets(data)
        (data,) = torch_geometric.loader.DataLoader(datasets, batch_size=len(datasets))
        return data


INFERENCE_CONFIGS = [
    BenchmarkApproximateSufficiencyConfig,
    BenchmarkApproximateSufficiencyLikelihoodRatioConfig,
    BenchmarkExpertSummaryConfig,
    BenchmarkLinearPosteriorMeanConfig,
    BenchmarkMinimumConditionalEntropyConfig,
    BenchmarkNeuralConfig,
    BenchmarkPLSConfig,
    CoalescentExpertSummaryConfig,
    CoalescentLinearPosteriorMeanConfig,
    CoalescentMinimumConditionalEntropyConfig,
    CoalescentNeuralConfig,
    CoalescentPLSConfig,
    TreeKernelApproximateSufficiencyConfig,
    TreeKernelApproximateSufficiencyLikelihoodRatioConfig,
    TreeKernelExpertSummaryConfig,
    TreeKernelLinearPosteriorMeanConfig,
    TreeKernelMinimumConditionalEntropyConfig,
    TreeKernelNeuralConfig,
    TreeKernelPLSConfig,
    PriorConfig,
]
INFERENCE_CONFIGS = {config.__name__: config for config in INFERENCE_CONFIGS}


def _build_pipeline(
    config: InferenceConfig, observed_data: Any | None = None
) -> Pipeline:
    transformer = config.create_transformer(observed_data=observed_data)
    return Pipeline(
        [
            ("transform", transformer),
            ("standardize", StandardScaler()),
            ("sample", NearestNeighborAlgorithm(n_samples=config.n_samples)),
        ]
    )


def __main__(argv: List[str] | None = None) -> None:
    start = datetime.now()
    parser = ArgumentParser("infer")
    parser.add_argument(
        "--n-samples", type=int, help="number of posterior samples to draw"
    )
    parser.add_argument(
        "--transformer-kwargs",
        type=json.loads,
        default={},
        help="keyword arguments for the transformer encoded as json",
    )
    parser.add_argument(
        "config", help="inference configuration to run", choices=INFERENCE_CONFIGS
    )
    parser.add_argument(
        "simulated", help="path to simulated data and parameters", type=resolve_path
    )
    parser.add_argument(
        "observed", help="path to observed data and parameters", type=resolve_path
    )
    parser.add_argument("output", help="path to output file", type=resolve_path)
    args: Args = parser.parse_args(argv)

    # Load the data and get the configuration.
    with args.simulated.open("rb") as fp:
        simulated = pickle.load(fp)
    with args.observed.open("rb") as fp:
        observed = pickle.load(fp)
    config: InferenceConfig = INFERENCE_CONFIGS[args.config](args)

    # If there is a preprocessor, we fit it to the simulated data and then apply it to both
    # datasets. Such a preprocessor can evaluate candidate summary statistics, for example.
    if preprocessor := config.create_preprocessor():
        with Timer() as timer:
            preprocessor.fit(simulated["data"])
            simulated["data"] = preprocessor.transform(simulated["data"])
            observed["data"] = preprocessor.transform(observed["data"])
        print(f"applied preprocessor in {timer.duration:.1f} seconds")

    # If the transformer is data-dependent, we have to handle each observation independently. We can
    # process them as a batch otherwise.
    if config.IS_DATA_DEPENDENT:
        samples = []
        for observed_data in tqdm(observed["data"]):
            pipeline = _build_pipeline(config, observed_data)
            pipeline.fit(simulated["data"], simulated["params"])
            samples.append(pipeline.predict([observed_data])[0])
        samples = np.asarray(samples)
    else:
        pipeline = _build_pipeline(config)
        with no_grad():
            pipeline.fit(simulated["data"], simulated["params"])
            samples = pipeline.predict(observed["data"])

    with args.output.open("wb") as fp:
        pickle.dump(
            {
                "args": vars(args),
                "start": start,
                "end": datetime.now(),
                "samples": samples,
                "params": observed["params"],
            },
            fp,
        )


if __name__ == "__main__":
    __main__()
