from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
import itertools as it
import numpy as np
from pathlib import Path
import pickle
from torch import as_tensor, get_default_dtype, nn, no_grad, Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Callable, Dict, List, Optional, Type

from ..experiments.coal import CoalPosteriorMixtureDensityTransformer, CoalPosteriorMeanTransformer
from ..nn import NegLogProbLoss
from ..transformers import NeuralTransformer
from .base import resolve_path


class Args:
    device: str
    config: str
    train: Path
    validation: Path
    output: Path


@dataclass
class TrainConfig:
    loss: Callable[..., Tensor]
    transformer_cls: Type[NeuralTransformer]
    transformer_kwargs: Dict[str, Any] | None = None
    max_epochs: int | None = None


TRAIN_CONFIGS = {
    "coal-neural_posterior_mean": TrainConfig(
        nn.MSELoss(),
        CoalPosteriorMeanTransformer,
    ),
    "coal-mixture_density_network": TrainConfig(
        NegLogProbLoss(),
        CoalPosteriorMixtureDensityTransformer,
    )
}


def _create_data_loader_from_pickle(path: Path, **kwargs) -> DataLoader:
    with path.open("rb") as fp:
        result = pickle.load(fp)
    dtype = kwargs.pop("dtype", get_default_dtype())
    device = kwargs.pop("device", None)
    data = as_tensor(result["data"], dtype=dtype, device=device)
    params = as_tensor(result["params"], dtype=dtype, device=device)
    dataset = TensorDataset(data, params)
    return DataLoader(dataset, **kwargs)


def __main__(argv: Optional[List[str]] = None) -> None:
    start = datetime.now()
    parser = ArgumentParser("train_transformer")
    parser.add_argument("--device", default="cpu", help="device to train on")
    parser.add_argument("config", choices=TRAIN_CONFIGS, help="training configuration")
    parser.add_argument("train", type=resolve_path, help="path to training data")
    parser.add_argument("validation", type=resolve_path, help="path to validation data")
    parser.add_argument("output", type=resolve_path, help="path to output file")
    args: Args = parser.parse_args(argv)

    # Load the data into tensor datasets.
    data_loader_kwargs = {
        "batch_size": 256,
        "shuffle": True,
        "device": args.device,
    }
    train_loader = _create_data_loader_from_pickle(args.train, **data_loader_kwargs)
    validation_loader = _create_data_loader_from_pickle(args.validation, **data_loader_kwargs)

    config = TRAIN_CONFIGS[args.config]
    transformer = config.transformer_cls(**(config.transformer_kwargs or {})).to(args.device)

    # Run one pilot batch to initialize the lazy modules.
    data: Tensor
    for data, _ in train_loader:
        transformer(data)
        break

    # Run the training.
    optim = Adam(transformer.parameters(), 0.001)
    scheduler = ReduceLROnPlateau(optim, verbose=True)
    stop_patience = 2 * scheduler.patience
    n_stop_patience_digits = len(str(stop_patience))

    best_loss = np.inf
    n_bad_epochs = 0
    for epoch in it.count(1):
        # Run one epoch.
        sizes = []
        loss_values = []
        params: Tensor
        for data, params in train_loader:
            optim.zero_grad()
            output = transformer(data)
            loss_value = config.loss(output, params)
            loss_value.backward()
            optim.step()
            sizes.append(data.shape[0])
            loss_values.append(loss_value.item())

        train_loss = np.dot(sizes, loss_values) / np.sum(sizes)

        # Evaluate the validation loss and update the learning rate.
        sizes = []
        loss_values = []
        with no_grad():
            for data, params in validation_loader:
                output = transformer(data)
                loss_value = config.loss(output, params)
                sizes.append(data.shape[0])
                loss_values.append(loss_value)

        validation_loss = np.dot(sizes, loss_values) / np.sum(sizes)
        scheduler.step(validation_loss)

        # Break if we've reached the maximum number of epochs.
        if config.max_epochs and epoch == config.max_epochs:
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
            f"loss={validation_loss:.4f}",
            f"best_loss={best_loss:.4f}",
            f"bad_epochs={n_bad_epochs:{n_stop_patience_digits}d} / {stop_patience}",
        ]
        print(' '.join(parts))
        if n_bad_epochs == 2 * scheduler.patience:
            break

    with args.output.open("wb") as fp:
        pickle.dump({
            "args": vars(args),
            "start": start,
            "end": datetime.now(),
            "transformer": transformer,
        }, fp)


if __name__ == "__main__":
    __main__()
