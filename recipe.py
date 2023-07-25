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
TREE_ROOT = ROOT / "tree"


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


def train_transformer(category: str, config: str, splits: Dict[str, Path]) -> Path:
    """
    Train a single transformer.
    """
    dependencies = [splits["train"], splits["validation"]]
    transformer_target = ROOT / f"{category}/transformers/{config}.pkl"
    action = ["python", "-m", "summaries.scripts.train_transformer", config, *dependencies,
              transformer_target]
    create_task(f"{category}:train:{config}", dependencies=dependencies, action=action,
                targets=[transformer_target])
    return transformer_target


def train_coalescent_transformers(splits: Dict[str, Path]) -> Dict[str, Path]:
    """
    Train transformers for the coalescent dataset.
    """
    return {config: train_transformer("coalescent", config, splits) for config in TRAIN_CONFIGS
            if config.startswith("Coalescent")}


def infer_posterior(splits: Dict[str, Path], config: str, category: str,
                    transformer: Path | None = None, name: str | None = None) -> Path:
    dependencies = [splits["train"], splits["test"]]
    name = name or config
    if transformer:
        dependencies.append(transformer)
        kwargs = {"transformer": str(transformer)}
        name = f"{name}-{transformer.with_suffix('').name}"
    else:
        kwargs = {}

    posterior_target = ROOT / f"{category}/samples/{name}.pkl"
    action = [
        "python", "-m", "summaries.scripts.infer_posterior", "--transformer-kwargs",
        json.dumps(kwargs), config, *dependencies[:2], posterior_target,
    ]
    create_task(f"{category}:infer:{name}", dependencies=dependencies, targets=[posterior_target],
                action=action)
    return posterior_target


def infer_mdn_posterior(splits: Dict[str, Path], category: str, transformer: Path,
                        loader: str | None = None) -> Path:
    dependencies = [transformer, splits["test"]]
    name = f"{category}:infer:{transformer.with_suffix('').name}"
    target = ROOT / f"{category}/samples/mdn-{transformer.name}"
    action = ["python", "-m", "summaries.scripts.infer_mdn", transformer, splits["test"], target]
    if loader:
        action.append(f"--loader={loader}")
    create_task(name, dependencies=dependencies, targets=[target], action=action)
    return target


def create_coalescent_tasks() -> Dict[str, Path]:
    splits = prepare_coalescent_data()

    transformers = train_coalescent_transformers(splits)
    sample_targets = {
        config: infer_posterior(splits, "CoalescentNeuralConfig", "coalescent", transformer) for
        config, transformer in transformers.items()
    }
    sample_targets |= {
        config: infer_posterior(splits, config, "coalescent") for config in INFERENCE_CONFIGS if
        config.startswith("Coalescent") and config != "CoalescentNeuralConfig"
    }
    sample_targets |= {
        "CoalescentMixtureDensityConfig": infer_mdn_posterior(
            splits, "coalescent", transformers["CoalescentMixtureDensityConfig"]
        ),
    }
    return sample_targets


def simulate_tree_data(folder: str, n_observations: int) -> Dict[str, Path]:
    data_root = ROOT / f"tree-{folder}" / "data"

    split_paths = {}
    splits = {"train": (100_000, 0), "validation": (1_000, 1), "test": (1_000, 2), "debug": (10, 3)}
    for split, (n_samples, seed) in splits.items():
        target = data_root / f"{split}.pkl"
        action = [
            "python", "-m", "summaries.scripts.simulate_data", f"--n-samples={n_samples}",
            f"--seed={seed}", f"--n-observations={n_observations}", "TreeSimulationConfig", target
        ]
        create_task(f"tree-{folder}:data:{split}", targets=[target], action=action)
        split_paths[split] = target
    return split_paths


def infer_tree_posterior_with_history_sampler(folder: str, splits: Dict[str, Path]) -> Path:
    config = "TreeKernelHistorySamplerConfig"
    posterior_target = ROOT / f"tree-{folder}/samples/{config}.pkl"
    action = [
        "python", "-m", "summaries.scripts.infer_tree_posterior", "--n-samples=1000",
        splits["test"], posterior_target,
    ]
    create_task(f"tree-{folder}:infer:{config}", dependencies=[splits["test"]],
                targets=[posterior_target], action=action)
    return posterior_target


def train_tree_transformers(folder: str, splits: Dict[str, Path]) -> Dict[str, Path]:
    """
    Train transformers for the tree dataset.
    """
    targets = {}
    for config in TRAIN_CONFIGS:
        if not config.startswith("Tree"):
            continue
        targets[config] = train_transformer(f"tree-{folder}", config, splits)
    return targets


def create_tree_tasks(folder: str, n_observations: int) -> None:
    splits = simulate_tree_data(folder, n_observations)
    transformers = train_tree_transformers(folder, splits)
    samples = {
        config: infer_posterior(splits, "TreeKernelNeuralConfig", f"tree-{folder}", transformer) for
        config, transformer in transformers.items()
    }
    samples |= {
        "TreeKernelHistorySamplerConfig": infer_tree_posterior_with_history_sampler(folder, splits),
    }
    samples |= {
        config: infer_posterior(splits, config, f"tree-{folder}") for config in INFERENCE_CONFIGS if
        config.startswith("Tree") and config != "TreeKernelNeuralConfig"
    }
    return samples


def simulate_benchmark_data(folder: str, n_observations: int) -> Dict[str, Path]:
    data_root = ROOT / f"benchmark-{folder}" / "data"

    split_paths = {}
    splits = {"train": (1_000_000, 0), "validation": (10_000, 1), "test": (1_000, 2),
              "debug": (10, 3)}
    for split, (n_samples, seed) in splits.items():
        target = data_root / f"{split}.pkl"
        action = ["python", "-m", "summaries.scripts.simulate_data", f"--n-samples={n_samples}",
                  f"--seed={seed}", f"--n-observations={n_observations}",
                  "BenchmarkSimulationConfig", target]
        create_task(f"benchmark-{folder}:data:{split}", targets=[target], action=action)
        split_paths[split] = target
    return split_paths


def train_benchmark_transformers(folder: str, splits: Dict[str, Path]) -> Dict[str, Path]:
    """
    Train transformers for the benchmark dataset.
    """
    targets = {}
    for config in TRAIN_CONFIGS:
        if not config.startswith("Benchmark"):
            continue
        targets[config] = train_transformer(f"benchmark-{folder}", config, splits)
    return targets


def create_benchmark_tasks() -> None:
    small_splits = simulate_benchmark_data("small", 10)
    large_splits = simulate_benchmark_data("large", 100)

    transformers = train_benchmark_transformers("small", small_splits)
    samples_small = {
        config: infer_posterior(small_splits, "BenchmarkNeuralConfig", "benchmark-small",
                                transformer) for config, transformer in transformers.items()
    }
    samples_small |= {
        config: infer_posterior(small_splits, config, "benchmark-small") for config in
        INFERENCE_CONFIGS if config.startswith("Benchmark") and config != "BenchmarkNeuralConfig"
    }

    # Train a mixture density network on the large dataset.
    mdn_compressor_large = train_transformer("benchmark-large", "BenchmarkMixtureDensityConfig",
                                             large_splits)
    reduced_mdn_compressor_large = train_transformer(
        "benchmark-large", "BenchmarkMixtureDensityConfigReduced", large_splits
    )

    # Add the Stan likelihood-based sampler.
    target = ROOT / "benchmark-small" / "samples" / "BenchmarkStanConfig.pkl"
    action = ["python", "-m", "summaries.scripts.infer_benchmark", small_splits["test"], target]
    samples_small["BenchmarkStanConfig"] = create_task(
        "benchmark-small:infer:stan", dependencies=[small_splits["test"]], targets=[target],
        action=action
    )

    target = ROOT / "benchmark-large" / "samples" / "BenchmarkStanConfig.pkl"
    action = ["python", "-m", "summaries.scripts.infer_benchmark", large_splits["test"], target]
    samples_small["BenchmarkStanConfig"] = create_task(
        "benchmark-large:infer:stan", dependencies=[large_splits["test"]], targets=[target],
        action=action
    )

    samples_small["BenchmarkMixtureDensityConfig"] = infer_mdn_posterior(
        small_splits, "benchmark-small", transformers["BenchmarkMixtureDensityConfig"]
    )
    samples_small["BenchmarkMixtureDensityConfigReduced"] = infer_mdn_posterior(
        small_splits, "benchmark-small", transformers["BenchmarkMixtureDensityConfigReduced"]
    )

    # Mixture density using the neural compressor on the large dataset.
    samples_large = {
        "BenchmarkNeuralConfig-large": infer_posterior(
            large_splits, "BenchmarkNeuralConfig", "benchmark-large",
            transformer=mdn_compressor_large, name="BenchmarkNeuralConfig-large",
        ),
        "BenchmarkNeuralConfig-large-reduced": infer_posterior(
            large_splits, "BenchmarkNeuralConfig", "benchmark-large",
            transformer=reduced_mdn_compressor_large, name="BenchmarkNeuralConfig-large",
        ),
        "BenchmarkNeuralConfig-small": infer_posterior(
            large_splits, "BenchmarkNeuralConfig", "benchmark-large",
            transformer=transformers["BenchmarkMixtureDensityConfig"],
            name="BenchmarkNeuralConfig-small",
        ),
        "BenchmarkNeuralConfig-small-reduced": infer_posterior(
            large_splits, "BenchmarkNeuralConfig", "benchmark-large",
            transformer=transformers["BenchmarkMixtureDensityConfigReduced"],
            name="BenchmarkNeuralConfig-small",
        ),
    }

    return samples_small, samples_large


create_benchmark_tasks()
create_coalescent_tasks()
create_tree_tasks("large", 748)
create_tree_tasks("small", 100)
