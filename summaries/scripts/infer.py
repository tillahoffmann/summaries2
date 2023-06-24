from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
import json
import numpy as np
from pathlib import Path
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List, Optional, Type

from ..algorithm import NearestNeighborAlgorithm
from ..transformers import as_transformer, MinimumConditionalEntropyTransformer, Transformer
from ..transformers.base import _DataDependentTransformerMixin


class Args:
    transformer: str
    simulated: Path
    observed: Path
    output: Path
    transformer_kwargs: Dict[str, Any]


def _resolved_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


@dataclass
class InferenceConfig:
    """
    Configuration for reproducible inference.
    """
    transformer_cls: Type[Transformer]
    frac: float
    transformer_kwargs: Optional[Dict[str, Any]] = None
    preprocessor_cls: Optional[Type[Transformer]] = None


INFERENCE_CONFIGS = {
    "coal-linear_posterior_mean": InferenceConfig(0.01, as_transformer(LinearRegression)),
    "coal-nonlinear_posterior_mean": InferenceConfig(0.01, as_transformer(MLPRegressor)),
    "coal-minimum_conditional_entropy": InferenceConfig(0.01, MinimumConditionalEntropyTransformer),
}


def _build_pipeline(config: InferenceConfig, **transformer_kwargs: Any) -> Pipeline:
    transformer = config.transformer_cls(**config.transformer_kwargs, **transformer_kwargs)
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
                        type=_resolved_path)
    parser.add_argument("observed", help="path to observed data and parameters",
                        type=_resolved_path)
    parser.add_argument("output", help="path to output file", type=_resolved_path)
    args: Args = parser.parse_args(argv)

    # Load the data and get the configuration.
    with args.simulated.open("rb") as fp:
        simulated = pickle.load(fp)
    with args.observed.open("rb") as fp:
        observed = pickle.load(fp)
    config = INFERENCE_CONFIGS[args.config]

    # If there is a preprocessor, we fit it to the simulated data and then apply it to both
    # datasets. Such a preprocessor can evaluate candidate summary statistics, for example.
    if config.preprocessor_cls:
        preprocessor = config.preprocessor_cls()
        preprocessor.fit(simulated["data"])
        simulated["data"] = preprocessor.transform(simulated["data"])
        observed["data"] = preprocessor.transform(observed["data"])

    # If the transformer is data-dependent, we have to handle each observation independently. We can
    # process them as a batch otherwise.
    if issubclass(config.transformer_cls, _DataDependentTransformerMixin):
        samples = []
        for observed_data in observed["data"]:
            pipeline = _build_pipeline(config, observed_data=observed_data,
                                       **args.transformer_kwargs)
            pipeline.fit(simulated["data"], simulated["params"])
            samples.append(pipeline.predict([observed_data])[0])
        samples = np.asarray(samples)
    else:
        pipeline = _build_pipeline(config, **args.transformer_kwargs)
        pipeline.fit(simulated["data"], simulated["params"])
        samples = pipeline.predict(observed["data"])

    with args.output.open("wb") as fp:
        pickle.dump({
            "args": vars(args),
            "start": start,
            "end": datetime.now(),
            "samples": samples,
        }, fp)


if __name__ == "__main__":
    __main__()
