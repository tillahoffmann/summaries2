from cook import create_task
import json
from pathlib import Path
from summaries.scripts.train_transformer import TRAIN_CONFIGS


create_task("requirements", action="pip-compile -v", targets=["requirements.txt"],
            dependencies=["requirements.in", "setup.py"])

create_task("tests", action="pytest -v --cov=summaries --cov-report=html --cov-fail-under=100")

# Download and extract the coalescent dataset.
root = Path("workspace/coal")
data_root = root / "data"
url = "https://github.com/tillahoffmann/coaloracle/releases/download/0.2/csv.zip"
archive = data_root / "coal.zip"
create_task("coal:download", action=f"curl -Lo {archive} {url}", targets=[archive])

coaloracle = data_root / "coaloracle.csv"
create_task("coal:extract", dependencies=[archive], action=f"unzip -ojd {data_root} {archive}",
            targets=[coaloracle])

# Preprocess the dataset by splitting it train, test, and validation sets.
splits = {"test": 1_000, "validation": 10_000, "train": 989_000}
split_targets = {split: data_root / f"{split}.pkl" for split in splits}
split_args = ' '.join(f"{split}:{size}" for split, size in splits.items())
action = f"python -m summaries.scripts.preprocess_coal --seed={21} {coaloracle} {data_root} " \
    f"{split_args}"
create_task("coal:preprocess", dependencies=[coaloracle], targets=split_targets.values(),
            action=action)

# Train the two compressors and run posterior inference.
for config in [config for config in TRAIN_CONFIGS if config.startswith("coal")]:
    dependencies = [split_targets["train"], split_targets["validation"]]
    transformer_target = root / f"transformers/{config}.pkl"
    action = ["python", "-m", "summaries.scripts.train_transformer", config, *dependencies,
              transformer_target]
    create_task(f"coal:train:{config}", dependencies=dependencies, action=action,
                targets=[transformer_target])

    dependencies = [split_targets["train"], split_targets["test"], transformer_target]
    posterior_target = root / f"samples/{config}.pkl"
    kwargs = {"transformer": str(transformer_target)}
    action = [
        "python", "-m", "summaries.scripts.infer", "--transformer-kwargs", json.dumps(kwargs),
        "coal-neural", *dependencies[:2], posterior_target,
    ]
    create_task(f"coal:infer:{config}", dependencies=dependencies, targets=[posterior_target],
                action=action)


config = "coal-linear_posterior_mean"
dependencies = [split_targets["train"], split_targets["test"]]
posterior_target = root / f"samples/{config}.pkl"
action = ["python", "-m", "summaries.scripts.infer", config, *dependencies, posterior_target]
create_task(f"coal:infer:{config}", dependencies=dependencies, targets=[posterior_target],
            action=action)
