from pathlib import Path
import pickle
from summaries.scripts.preprocess_coal import __main__


def test_preprocess_coal(tmp_path: Path) -> None:
    coaloracle = Path(__file__).parent.parent / "data/coaloracle_sample.csv"
    splits = {"foo": 3, "bar": 6}
    argv = [coaloracle, tmp_path, *[f"{split}:{size}" for split, size in splits.items()]]
    __main__(map(str, argv))

    for split, size in splits.items():
        with (tmp_path / f"{split}.pkl").open("rb") as fp:
            result = pickle.load(fp)
            assert result["data"].shape == (size, 7)
            assert result["params"].shape == (size, 2)
