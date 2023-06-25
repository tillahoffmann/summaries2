from argparse import ArgumentParser
from datetime import datetime
import json
import numpy as np
from pathlib import Path
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import no_grad
from typing import Any, Dict, List, Optional

from ..algorithm import NearestNeighborAlgorithm
from ..transformers import MinimumConditionalEntropyTransformer, PredictorTransformer, Transformer
from .base import resolve_path


class Args:
    config: str
    simulated: Path
    observed: Path
    output: Path
    transformer_kwargs: Dict[str, Any]


class InferenceConfig:
    """
    Base class for inference configurations.
    """
    def __init__(self, frac: float, is_data_dependent: bool,
                 preprocessor: Transformer | None = None) -> None:
        self.frac = frac
        self.is_data_dependent = is_data_dependent
        self.preprocessor = preprocessor

    def create_transformer(self, args: Args, observed_data: Any | None = None) -> Transformer:
        raise NotImplementedError


class CoalescentConfig(InferenceConfig):
    def __init__(self, is_data_dependent: bool = False) -> None:
        super().__init__(0.01, is_data_dependent)


class CoalescentMinimumConditionalEntropyConfig(CoalescentConfig):
    def __init__(self) -> None:
        super().__init__(True)

    def create_transformer(self, args: Args, observed_data: np.ndarray) -> Transformer:
        return MinimumConditionalEntropyTransformer(observed_data, self.frac)


class CoalescentLinearPosteriorMeanConfig(CoalescentConfig):
    def create_transformer(self, args: Args, observed_data: Any | None = None) -> Transformer:
        return PredictorTransformer(LinearRegression())


class CoalescentNeuralConfig(CoalescentConfig):
    def create_transformer(self, args: Args, observed_data: Any | None = None) -> Transformer:
        with open(args.transformer_kwargs["transformer"], "rb") as fp:
            return pickle.load(fp)["transformer"]


INFERENCE_CONFIGS = [
    CoalescentLinearPosteriorMeanConfig(),
    CoalescentMinimumConditionalEntropyConfig(),
    CoalescentNeuralConfig(),
]
INFERENCE_CONFIGS = {config.__class__.__name__: config for config in INFERENCE_CONFIGS}


def _build_pipeline(args: Args, config: InferenceConfig, observed_data: Any | None = None) \
        -> Pipeline:
    transformer = config.create_transformer(args, observed_data=observed_data)
    return Pipeline([
        ("transform", transformer),
        ("standardize", StandardScaler()),
        ("sample", NearestNeighborAlgorithm(config.frac)),
    ])


def __main__(argv: Optional[List[str]] = None) -> None:
    start = datetime.now()
    parser = ArgumentParser("infer")
    parser.add_argument("--transformer-kwargs", type=json.loads, default={},
                        help="keyword arguments for the transformer encoded as json")
    parser.add_argument("config", help="inference configuration to run", choices=INFERENCE_CONFIGS)
    parser.add_argument("simulated", help="path to simulated data and parameters",
                        type=resolve_path)
    parser.add_argument("observed", help="path to observed data and parameters",
                        type=resolve_path)
    parser.add_argument("output", help="path to output file", type=resolve_path)
    args: Args = parser.parse_args(argv)

    # Load the data and get the configuration.
    with args.simulated.open("rb") as fp:
        simulated = pickle.load(fp)
    with args.observed.open("rb") as fp:
        observed = pickle.load(fp)
    config: InferenceConfig = INFERENCE_CONFIGS[args.config]

    # If there is a preprocessor, we fit it to the simulated data and then apply it to both
    # datasets. Such a preprocessor can evaluate candidate summary statistics, for example.
    if config.preprocessor:
        config.preprocessor.fit(simulated["data"])
        simulated["data"] = config.preprocessor.transform(simulated["data"])
        observed["data"] = config.preprocessor.transform(observed["data"])

    # If the transformer is data-dependent, we have to handle each observation independently. We can
    # process them as a batch otherwise.
    if config.is_data_dependent:
        samples = []
        for observed_data in observed["data"]:
            pipeline = _build_pipeline(args, config, observed_data)
            pipeline.fit(simulated["data"], simulated["params"])
            samples.append(pipeline.predict([observed_data])[0])
        samples = np.asarray(samples)
    else:
        pipeline = _build_pipeline(args, config)
        with no_grad():
            pipeline.fit(simulated["data"], simulated["params"])
            samples = pipeline.predict(observed["data"])

    with args.output.open("wb") as fp:
        pickle.dump({
            "args": vars(args),
            "start": start,
            "end": datetime.now(),
            "samples": samples,
            "pipeline": pipeline,
        }, fp)


if __name__ == "__main__":
    __main__()
