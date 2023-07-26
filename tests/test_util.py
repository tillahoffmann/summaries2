from pathlib import Path
from summaries import util


def test_pickle_unpickle(tmp_path: Path) -> None:
    obj = {
        "a": None,
        "b": 17,
        "c": [1, 2, "asdf"],
    }
    path = tmp_path / "test.pkl"
    util.dump_pickle(obj, path)
    assert util.load_pickle(path) == obj
