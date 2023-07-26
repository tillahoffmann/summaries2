from pathlib import Path
import pickle
from typing import Any


def load_pickle(path: str | Path) -> Any:
    """
    Load a pickled file.
    """
    try:
        with open(path, "rb") as fp:
            return pickle.load(fp)
    except Exception as ex:
        raise pickle.UnpicklingError(f"failed to unpickle '{path}'") from ex


def dump_pickle(obj: Any, path: str | Path) -> None:
    """
    Dump an object to a pickle file.
    """
    try:
        with open(path, "wb") as fp:
            pickle.dump(obj, fp)
    except Exception as ex:
        raise pickle.PicklingError(f"failed to pickle object to '{path}'") from ex
