from cook import create_task
import json
from pathlib import Path
from summaries.scripts.infer_posterior import INFERENCE_CONFIGS
from summaries.scripts.train_transformer import TRAIN_CONFIGS
from typing import Dict


create_task("requirements", action="pip-compile -v", targets=["requirements.txt"],
            dependencies=["requirements.in", "setup.py"])

create_task("tests", action="pytest -v --cov=summaries --cov-report=html --cov-fail-under=100")


ROOT = Path("workspace")
COALESCENT_ROOT = ROOT / "coalescent"
GRAPH_ROOT = ROOT / "graph"


def prepare_coalescent_data() -> Dict[str, Path]:
    """
    Download, extract, and preprocess the coalescent dataset.
    """
    data_root = COALESCENT_ROOT / "data"
    url = "https://github.com/tillahoffmann/coaloracle/releases/download/0.2/csv.zip"
    archive = data_root / "coal.zip"
    create_task("coalescent:download", action=f"curl -Lo {archive} {url}", targets=[archive])

    coaloracle = data_root / "coaloracle.csv"
    create_task("coalescent:extract", dependencies=[archive], targets=[coaloracle],
                action=f"unzip -ojd {data_root} {archive}")

    # Preprocess the dataset by splitting it train, test, and validation sets.
    splits = {"test": 1_000, "validation": 10_000, "train": 989_000}
    split_targets = {split: data_root / f"{split}.pkl" for split in splits}
    split_args = ' '.join(f"{split}:{size}" for split, size in splits.items())
    action = f"python -m summaries.scripts.preprocess_coalescent --seed={21} {coaloracle} " \
        f"{data_root} {split_args}"
    create_task("coalescent:preprocess", dependencies=[coaloracle], targets=split_targets.values(),
                action=action)

    return split_targets


def train_coalescent_transformers(splits: Dict[str, Path]) -> Dict[str, Path]:
    """
    Train transformers for the coalescent dataset.
    """
    dependencies = [splits["train"], splits["validation"]]
    targets = {}
    for config in TRAIN_CONFIGS:
        if not config.startswith("Coalescent"):
            continue

        transformer_target = COALESCENT_ROOT / f"transformers/{config}.pkl"
        action = ["python", "-m", "summaries.scripts.train_transformer", config, *dependencies,
                  transformer_target]
        create_task(f"coalescent:train:{config}", dependencies=dependencies, action=action,
                    targets=[transformer_target])

        targets[config] = transformer_target
    return targets


def infer_coalescent_posterior(splits: Dict[str, Path], config: str,
                               transformer: Path | None = None) -> None:
    dependencies = [splits["train"], splits["test"]]
    name = config
    if transformer:
        dependencies.append(transformer)
        kwargs = {"transformer": str(transformer)}
        name = f"{config}-{transformer.with_suffix('').name}"
    else:
        kwargs = {}

    posterior_target = COALESCENT_ROOT / f"samples/{name}.pkl"
    action = [
        "python", "-m", "summaries.scripts.infer", "--transformer-kwargs", json.dumps(kwargs),
        config, *dependencies[:2], posterior_target,
    ]
    create_task(f"coalescent:infer:{name}", dependencies=dependencies, targets=[posterior_target],
                action=action)
    return posterior_target


def create_coalescent_tasks() -> Dict[str, Path]:
    splits = prepare_coalescent_data()

    transformers = train_coalescent_transformers(splits)
    sample_targets = {
        config: infer_coalescent_posterior(splits, "CoalescentNeuralConfig", transformer) for
        config, transformer in transformers.items()
    }
    sample_targets |= {
        config: infer_coalescent_posterior(splits, config) for config in INFERENCE_CONFIGS if
        config.startswith("Coalescent") and config != "CoalescentNeuralConfig"
    }
    return sample_targets


def simulate_graph_data() -> Dict[str, Path]:
    data_root = GRAPH_ROOT / "data"

    splits = {"train": (100_000, 0), "validation": (1_000, 1), "test": (1_000, 2)}
    for split, (n_samples, seed) in splits.items():
        target = data_root / f"{split}.pkl"
        action = ["python", "-m", "summaries.scripts.simulate_data", f"--n-samples={n_samples}",
                  f"--seed={seed}", "GraphSimulationConfig", target]
        create_task(f"graph:data:{split}", targets=[target], action=action)


def create_graph_tasks() -> None:
    simulate_graph_data()


create_coalescent_tasks()
create_graph_tasks()
