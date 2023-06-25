from cook import create_task
from pathlib import Path


create_task("requirements", action="pip-compile -v", targets=["requirements.txt"],
            dependencies=["requirements.in", "setup.py"])

create_task("tests", action="pytest -v --cov=summaries --cov-report=html --cov-fail-under=100")

# Download and extract the coalescent dataset.
root = Path("workspace/data/coal")
url = "https://github.com/tillahoffmann/coaloracle/releases/download/0.2/csv.zip"
archive = root / "coal.zip"
create_task("coal:download", action=f"curl -Lo {archive} {url}", targets=[archive])

coaloracle = root / "coaloracle.csv"
create_task("coal:extract", dependencies=[archive], action=f"unzip -jd {root} {archive}",
            targets=[coaloracle])

# Preporcess the dataset by splitting it train, test, and validation sets.
splits = {"test": 1_000, "validation": 10_000, "train": 989_000}
split_targets = {split: root / f"{split}.pkl" for split in splits}
split_args = ' '.join(f"{split}:{size}" for split, size in splits.items())
create_task("coal:preprocess", dependencies=[coaloracle], targets=split_targets.values(),
            action=f"python -m summaries.scripts.preprocess_coal {coaloracle} {root} {split_args}")
