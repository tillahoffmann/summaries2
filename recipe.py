from cook import create_task, Task
import json
from pathlib import Path
import shutil
from summaries.scripts.infer_posterior import INFERENCE_CONFIGS
from summaries.scripts.train_transformer import TRAIN_CONFIGS
from summaries.util import load_pickle
from typing import Dict


create_task("requirements", action="pip-compile -v", targets=["requirements.txt"],
            dependencies=["requirements.in", "setup.py"])

create_task("tests", action="pytest -v --cov=summaries --cov-report=html --cov-report=term-missing "
            "--cov-fail-under=100")


ROOT = Path("workspace")
BENCHMARK_ROOT = ROOT / "benchmark"
COALESCENT_ROOT = ROOT / "coalescent"
TREE_ROOT = ROOT / "tree"
# Random number generator seeds generated, at some point, by np.random.randint(10_000).
SEEDS = [6389, 9074, 7627]


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


def _pick_best_transformer(task: Task) -> None:
    # Find the best one and copy it; a symlink would be nice but that breaks all sorts of stuff,
    # e.g., evaluating digests.
    best = min(task.dependencies, key=lambda path: load_pickle(path)["last_validation_loss"])
    shutil.copy(best, task.targets[0])


def train_transformer(experiment: str, splits: Dict[str, Path], config: str) -> Path:
    """
    Train a single transformer multiple times using different seeds. We create a symlink to the
    "best" transformer as evaluated by the last validation loss of the training run.

    Args:
        experiment: Parent folder of the experiment.
        splits: Datasets for training and validation.
        config: Training configuration supported by `summaries.scripts.train_transformer`.

    Returns:
        Path to trained transformer.
    """
    dependencies = [splits["train"], splits["validation"]]
    name = f"{experiment}:train:{config}"

    # Train a bunch of transformers with different seeds.
    transformer_targets = []
    for seed in SEEDS:
        transformer_target = ROOT / f"{experiment}/transformers/{config}-{seed}.pkl"
        action = ["python", "-m", "summaries.scripts.train_transformer", f"--seed={seed}", config,
                  *dependencies, transformer_target]
        create_task(f"{name}-{seed}", dependencies=dependencies, action=action,
                    targets=[transformer_target])
        transformer_targets.append(transformer_target)

    # Pick the best one using the validation loss.
    transformer_target = ROOT / f"{experiment}/transformers/{config}.pkl"
    create_task(name, action=_pick_best_transformer, targets=[transformer_target],
                dependencies=transformer_targets)

    return transformer_target


def train_coalescent_transformers(splits: Dict[str, Path]) -> Dict[str, Path]:
    """
    Train transformers for the coalescent dataset.
    """
    return {config: train_transformer("coalescent", splits, config) for config in TRAIN_CONFIGS
            if config.startswith("Coalescent")}


def infer_posterior(experiment: str, splits: Dict[str, Path], config: str,
                    transformer: Path | None = None, suffix: str | None = None) -> Path:
    """
    Draw posterior samples using approximate Bayesian computation.

    Args:
        experiment: Parent folder of the experiment.
        splits: Datasets for training and validation.
        config: Training configuration supported by `summaries.scripts.infer_posterior`.
        transformer: Optional path to pickled transformer.
        suffix: Suffix for file comprising samples (defaults to transformer name if available).

    Returns:
        Path to file comprising posterior samples.
    """
    dependencies = [splits["train"], splits["test"]]
    if transformer:
        dependencies.append(transformer)
        kwargs = {"transformer": str(transformer)}
        suffix = suffix or transformer.with_suffix('').name
    else:
        kwargs = {}

    name = f"{config}-{suffix}" if suffix else config
    posterior_target = ROOT / f"{experiment}/samples/{name}.pkl"
    action = [
        "python", "-m", "summaries.scripts.infer_posterior", "--transformer-kwargs",
        json.dumps(kwargs), config, *dependencies[:2], posterior_target,
    ]
    create_task(f"{experiment}:infer:{name}", dependencies=dependencies, targets=[posterior_target],
                action=action)
    return posterior_target


def infer_mdn_posterior(experiment: str, splits: Dict[str, Path], transformer: Path,
                        loader: str | None = None) -> Path:
    """
    Infer posterior samples by sampling from a mixture density network.

    Args:
        experiment: Parent folder of the experiment.
        splits: Datasets for training and validation.
        transformer: Path to pickled mixture density network.
        loader: Name of the data loader (required for loading graph data).
    """
    dependencies = [transformer, splits["test"]]
    name = f"{experiment}:infer:{transformer.with_suffix('').name}"
    target = ROOT / f"{experiment}/samples/mdn-{transformer.name}"
    action = ["python", "-m", "summaries.scripts.infer_mdn", transformer, splits["test"], target]
    if loader:
        action.append(f"--loader={loader}")
    create_task(name, dependencies=dependencies, targets=[target], action=action)
    return target


def create_coalescent_tasks() -> Dict[str, Path]:
    """
    Create all tasks for the coalescent experiment.

    Returns:
        Map from names to paths comprising samples.
    """
    splits = prepare_coalescent_data()

    transformers = train_coalescent_transformers(splits)
    sample_targets = {
        f"CoalescentNeuralConfig-{config}":
            infer_posterior("coalescent", splits, "CoalescentNeuralConfig", transformer) for
            config, transformer in transformers.items()
    } | {
        config: infer_posterior("coalescent", splits, config) for config in INFERENCE_CONFIGS if
        config.startswith("Coalescent") and config != "CoalescentNeuralConfig"
    } | {
        "CoalescentMixtureDensityConfig": infer_mdn_posterior(
            "coalescent", splits, transformers["CoalescentMixtureDensityConfig"]
        ),
        "PriorConfig": infer_posterior("coalescent", splits, "PriorConfig")
    }
    return {
        "samples": sample_targets,
        "experiment": "coalescent",
        "transformers": transformers,
    }


def simulate_tree_data(experiment: str, n_observations: int) -> Dict[str, Path]:
    """
    Simulate data from a growing tree.

    Args:
        experiment: Parent folder of the experiment.
        n_observations: Number of nodes per graph.

    Returns:
        Dataset splits.
    """
    data_root = ROOT / experiment / "data"

    split_paths = {}
    splits = {"train": (100_000, 0), "validation": (1_000, 1), "test": (1_000, 2), "debug": (10, 3)}
    for split, (n_samples, seed) in splits.items():
        target = data_root / f"{split}.pkl"
        action = [
            "python", "-m", "summaries.scripts.simulate_data", f"--n-samples={n_samples}",
            f"--seed={seed}", f"--n-observations={n_observations}", "TreeSimulationConfig", target
        ]
        create_task(f"{experiment}:data:{split}", targets=[target], action=action)
        split_paths[split] = target
    return split_paths


def infer_tree_posterior_with_history_sampler(experiment: str, splits: Dict[str, Path]) -> Path:
    """
    Infer tree posteriors using the history sampler of https://github.com/gstonge/fasttr.
    """
    config = "TreeKernelHistorySamplerConfig"
    posterior_target = ROOT / f"{experiment}/samples/{config}.pkl"
    action = [
        "python", "-m", "summaries.scripts.infer_tree_posterior", "--n-samples=1000",
        splits["test"], posterior_target,
    ]
    create_task(f"{experiment}:infer:{config}", dependencies=[splits["test"]],
                targets=[posterior_target], action=action)
    return posterior_target


def train_tree_transformers(experiment: str, splits: Dict[str, Path]) -> Dict[str, Path]:
    """
    Train transformers for the tree dataset.
    """
    targets = {}
    for config in TRAIN_CONFIGS:
        if not config.startswith("Tree"):
            continue
        targets[config] = train_transformer(experiment, splits, config)
    return targets


def create_tree_tasks(experiment: str, n_observations: int) -> None:
    splits = simulate_tree_data(experiment, n_observations)
    transformers = train_tree_transformers(experiment, splits)
    samples = {
        f"TreeKernelNeuralConfig-{config}":
            infer_posterior(experiment, splits, "TreeKernelNeuralConfig", transformer) for
            config, transformer in transformers.items()
    } | {
        "TreeKernelHistorySamplerConfig":
            infer_tree_posterior_with_history_sampler(experiment, splits),
        "PriorConfig": infer_posterior(experiment, splits, "PriorConfig"),
    } | {
        config: infer_posterior(experiment, splits, config) for config in INFERENCE_CONFIGS if
        config.startswith("Tree") and config != "TreeKernelNeuralConfig"
    }
    return {
        "splits": splits,
        "transformers": transformers,
        "samples": samples,
        "experiment": experiment,
    }


def simulate_benchmark_data(experiment: str, n_observations: int) -> Dict[str, Path]:
    data_root = ROOT / experiment / "data"

    split_paths = {}
    splits = {"train": (1_000_000, 0), "validation": (10_000, 1), "test": (1_000, 2),
              "debug": (10, 3)}
    for split, (n_samples, seed) in splits.items():
        target = data_root / f"{split}.pkl"
        action = [
            "python", "-m", "summaries.scripts.simulate_data", f"--n-samples={n_samples}",
            f"--seed={seed}", f"--n-observations={n_observations}", "BenchmarkSimulationConfig",
            target
        ]
        create_task(f"{experiment}:data:{split}", targets=[target], action=action)
        split_paths[split] = target
    return split_paths


def train_benchmark_transformers(experiment: str, splits: Dict[str, Path]) -> Dict[str, Path]:
    """
    Train transformers for the benchmark dataset.
    """
    targets = {}
    for config in TRAIN_CONFIGS:
        if not config.startswith("Benchmark"):
            continue
        targets[config] = train_transformer(experiment, splits, config)
    return targets


def create_benchmark_tasks(experiment: str, n_observations: int) -> None:
    splits = simulate_benchmark_data(experiment, n_observations)
    transformers = train_benchmark_transformers(experiment, splits)
    samples = {
        f"BenchmarkNeuralConfig-{config}":
            infer_posterior(experiment, splits, "BenchmarkNeuralConfig", transformer) for
            config, transformer in transformers.items()
    } | {
        config: infer_posterior(experiment, splits, config) for config in
        INFERENCE_CONFIGS if config.startswith("Benchmark") and config != "BenchmarkNeuralConfig"
    } | {
        "PriorConfig": infer_posterior(experiment, splits, "PriorConfig"),
        "BenchmarkMixtureDensityConfig": infer_mdn_posterior(
            experiment, splits, transformers["BenchmarkMixtureDensityConfig"]
        ),
        "BenchmarkMixtureDensityConfigReduced": infer_mdn_posterior(
            experiment, splits, transformers["BenchmarkMixtureDensityConfigReduced"]
        ),
    }

    # Add the Stan likelihood-based sampler.
    target = ROOT / experiment / "samples" / "BenchmarkStanConfig.pkl"
    action = ["python", "-m", "summaries.scripts.infer_benchmark", splits["test"], target]
    create_task(
        f"{experiment}:infer:stan", dependencies=[splits["test"]], targets=[target],
        action=action
    )
    samples["BenchmarkStanConfig"] = target

    return {
        "splits": splits,
        "transformers": transformers,
        "samples": samples,
        "experiment": experiment,
    }


coalescent_tasks = create_coalescent_tasks()
benchmark_tasks_small = create_benchmark_tasks("benchmark-small", 10)
benchmark_tasks_large = create_benchmark_tasks("benchmark-large", 100)
tree_tasks_small = create_tree_tasks("tree-small", 100)
tree_tasks_large = create_tree_tasks("tree-large", 748)

# Create the transfer learning tasks: using the networks trained on smaller datasets for inference
# on the larger ones.
tree_tasks_large["samples"]["TreeKernelNeuralConfig-TreeMixtureDensityConfig-small"] = \
    infer_posterior("tree-large", tree_tasks_large["splits"], "TreeKernelNeuralConfig",
                    tree_tasks_small["transformers"]["TreeMixtureDensityConfig"],
                    suffix="TreeMixtureDensityConfig-small")
benchmark_tasks_large["samples"]["BenchmarkNeuralConfig-BenchmarkMixtureDensityConfig-small"] = \
    infer_posterior("benchmark-large", benchmark_tasks_large["splits"], "BenchmarkNeuralConfig",
                    benchmark_tasks_small["transformers"]["BenchmarkMixtureDensityConfig"],
                    suffix="BenchmarkMixtureDensityConfig-small")

# Add evaluation for each batch of tasks.
for tasks in [coalescent_tasks, benchmark_tasks_large, benchmark_tasks_small, tree_tasks_large,
              tree_tasks_small]:
    experiment = tasks["experiment"]
    target = ROOT / experiment / "evaluation.csv"
    paths = list(tasks["samples"].values())
    action = f"python -m summaries.scripts.evaluate --csv={target} {' '.join(map(str, paths))}"
    create_task(f"{experiment}:evaluation", targets=[target], action=action, dependencies=paths)
