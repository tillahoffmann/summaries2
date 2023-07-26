from pathlib import Path
import pickle
import pytest
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


def test_pickle_unpickle_fail() -> None:
    with pytest.raises(pickle.PicklingError, match="failed to pickle object"):
        util.dump_pickle({}, "/not/a/directory.pkl")
    with pytest.raises(pickle.UnpicklingError, match="failed to unpickle"):
        util.load_pickle("/not/a/directory.pkl")
