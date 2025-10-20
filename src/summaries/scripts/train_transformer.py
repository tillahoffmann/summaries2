from argparse import ArgumentParser
from datetime import datetime
import itertools as it
import numpy as np
from pathlib import Path
import pickle
from snippets.tensor_data_loader import TensorDataLoader
from torch import as_tensor, get_default_dtype, nn, no_grad, Tensor
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset
from torch_geometric.data import Data as GeometricData
from torch_geometric.loader import DataLoader as GeometricDataLoader
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..experiments.benchmark import (
    BenchmarkPosteriorMeanTransformer,
    BenchmarkPosteriorMixtureDensityTransformer,
)
from ..experiments.coalescent import (
    CoalescentPosteriorMixtureDensityTransformer,
    CoalescentPosteriorMeanTransformer,
)
from ..experiments.tree import (
    predecessors_to_datasets,
    TreePosteriorMeanTransformer,
    TreePosteriorMixtureDensityTransformer,
)
from ..nn import NegLogProbLoss
from .base import resolve_path


class Args:
    device: str
    config: str
    train: Path
    validation: Path
    output: Path
    seed: Optional[int]


class TrainConfig:
    """
    Base class for training configurations.
    """

    LOSS: Callable[..., Tensor] | None = None
    MAX_EPOCHS: int | None = None
    DATA_LOADER_KWARGS: Dict[str, Any] = {}

    def __init__(self, args: Args) -> None:
        self.args = args
        assert self.LOSS is not None

    def create_transformer(self):
        raise NotImplementedError

    def create_data_loader(self, path: Path, **kwargs: Any) -> TensorDataLoader:
        with path.open("rb") as fp:
            result = pickle.load(fp)
        kwargs = self.DATA_LOADER_KWARGS | kwargs
        dtype = kwargs.pop("dtype", get_default_dtype())
        device = kwargs.pop("device", None)
        data = as_tensor(result["data"], dtype=dtype, device=device)
        params = as_tensor(result["params"], dtype=dtype, device=device)
        dataset = TensorDataset(data, params)
        return TensorDataLoader(dataset, **kwargs)


class BenchmarkTrainConfig(TrainConfig):
    DATA_LOADER_KWARGS = {
        "batch_size": 512,
        "shuffle": True,
    }


class BenchmarkPosteriorMeanConfig(BenchmarkTrainConfig):
    LOSS = nn.MSELoss()

    def create_transformer(self):
        return BenchmarkPosteriorMeanTransformer()


class BenchmarkMixtureDensityConfig(BenchmarkTrainConfig):
    LOSS = NegLogProbLoss()

    def create_transformer(self):
        return BenchmarkPosteriorMixtureDensityTransformer()


class BenchmarkMixtureDensityConfigReduced(BenchmarkTrainConfig):
    LOSS = NegLogProbLoss()

    def create_transformer(self):
        return BenchmarkPosteriorMixtureDensityTransformer(2)


class CoalescentTrainConfig(TrainConfig):
    DATA_LOADER_KWARGS = {
        "batch_size": 256,
        "shuffle": True,
    }


class CoalescentPosteriorMeanConfig(CoalescentTrainConfig):
    LOSS = nn.MSELoss()

    def create_transformer(self):
        return CoalescentPosteriorMeanTransformer()


class CoalescentMixtureDensityConfig(CoalescentTrainConfig):
    LOSS = NegLogProbLoss()

    def create_transformer(self):
        return CoalescentPosteriorMixtureDensityTransformer()


class TreeTrainConfig(TrainConfig):
    DATA_LOADER_KWARGS = {
        "batch_size": 32,
        "shuffle": True,
    }

    def create_data_loader(self, path: Path, **kwargs: Any) -> GeometricDataLoader:
        with path.open("rb") as fp:
            result = pickle.load(fp)
        device = kwargs.pop("device", None)
        datasets = predecessors_to_datasets(result["data"], result["params"], device)
        return GeometricDataLoader(datasets, **self.DATA_LOADER_KWARGS)


class TreeMixtureDensityConfig(TreeTrainConfig):
    LOSS = NegLogProbLoss()

    def create_transformer(self):
        return TreePosteriorMixtureDensityTransformer()


class TreePosteriorMeanConfig(TreeTrainConfig):
    LOSS = nn.MSELoss()

    def create_transformer(self):
        return TreePosteriorMeanTransformer()


TRAIN_CONFIGS = [
    BenchmarkPosteriorMeanConfig,
    BenchmarkMixtureDensityConfig,
    BenchmarkMixtureDensityConfigReduced,
    CoalescentMixtureDensityConfig,
    CoalescentPosteriorMeanConfig,
    TreeMixtureDensityConfig,
    TreePosteriorMeanConfig,
]
TRAIN_CONFIGS = {config.__name__: config for config in TRAIN_CONFIGS}


def _expand_batch(
    batch: Tuple[torch.Tensor, torch.Tensor] | GeometricData,
) -> Tuple[torch.Tensor, torch.Tensor, int] | Tuple[GeometricData, int]:
    if isinstance(batch, GeometricData):
        data = batch
        params = batch.params
        n = batch.num_graphs
    else:
        data, params = batch
        n = data.shape[0]
    return data, params, n


def __main__(argv: Optional[List[str]] = None) -> None:
    start = datetime.now()
    parser = ArgumentParser("train_transformer")
    parser.add_argument("--device", default="cpu", help="device to train on")
    parser.add_argument("--seed", type=int, help="random number generator seed")
    parser.add_argument("config", choices=TRAIN_CONFIGS, help="training configuration")
    parser.add_argument("train", type=resolve_path, help="path to training data")
    parser.add_argument("validation", type=resolve_path, help="path to validation data")
    parser.add_argument("output", type=resolve_path, help="path to output file")
    args: Args = parser.parse_args(argv)

    # Set a seed for reproducibility.
    if args.seed is not None:
        torch.manual_seed(args.seed)
    config: TrainConfig = TRAIN_CONFIGS[args.config](args)

    # Load the data into tensor datasets.
    train_loader = config.create_data_loader(args.train, device=args.device)
    validation_loader = config.create_data_loader(args.validation, device=args.device)

    # Run one pilot batch to initialize the lazy modules.
    transformer = config.create_transformer().to(args.device)
    for batch in train_loader:
        data, *_ = _expand_batch(batch)
        transformer(data)
        break

    # Run the training.
    optim = Adam(transformer.parameters(), 0.01)
    scheduler = ReduceLROnPlateau(optim)
    stop_patience = 2 * scheduler.patience
    n_stop_patience_digits = len(str(stop_patience))

    best_loss = np.inf
    n_bad_epochs = 0
    for epoch in it.count(1):
        # Run one epoch.
        sizes = []
        loss_values = []
        params: Tensor
        for batch in train_loader:
            data, params, n = _expand_batch(batch)
            optim.zero_grad()
            output = transformer(data)
            loss_value = config.LOSS(output, params)
            loss_value.backward()
            optim.step()
            sizes.append(n)
            loss_values.append(loss_value.item())

        train_loss = np.dot(sizes, loss_values) / np.sum(sizes)

        # Evaluate the validation loss and update the learning rate.
        sizes = []
        loss_values = []
        with no_grad():
            for batch in validation_loader:
                data, params, n = _expand_batch(batch)
                output = transformer(data)
                loss_value = config.LOSS(output, params)
                sizes.append(n)
                loss_values.append(loss_value)

        validation_loss = np.dot(sizes, loss_values) / np.sum(sizes)
        scheduler.step(validation_loss)

        # Break if we've reached the maximum number of epochs.
        if config.MAX_EPOCHS and epoch == config.MAX_EPOCHS:
            break

        # Determine whether to stop training based on validation loss.
        if validation_loss + scheduler.threshold < best_loss:
            best_loss = validation_loss
            n_bad_epochs = 0
        else:
            n_bad_epochs += 1

        parts = [
            f"epoch={epoch}",
            f"train_loss={train_loss:.4f}",
            f"validation_loss={validation_loss:.4f}",
            f"best_loss={best_loss:.4f}",
            f"bad_epochs={n_bad_epochs:{n_stop_patience_digits}d} / {stop_patience}",
        ]
        print(" ".join(parts))
        if n_bad_epochs == 2 * scheduler.patience:
            break

    with args.output.open("wb") as fp:
        pickle.dump(
            {
                "args": vars(args),
                "start": start,
                "end": datetime.now(),
                "transformer": transformer,
                "last_validation_loss": validation_loss,
            },
            fp,
        )


if __name__ == "__main__":
    __main__()
