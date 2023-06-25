from pathlib import Path


def resolve_path(path: str | Path) -> Path:
    """
    Resolve an absolute path.
    """
    return Path(path).expanduser().resolve()
