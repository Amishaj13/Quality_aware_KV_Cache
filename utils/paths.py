"""Project-relative paths. Nothing here depends on the notebook's working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve(path) -> Path:
    """Resolve a config path relative to the project root (absolute paths kept)."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def results_dirs(results_directory="results"):
    root = resolve(results_directory)
    dirs = {
        "root": root,
        "raw_logs": root / "raw_logs",
        "json": root / "json",
        "csv": root / "csv",
        "charts": root / "charts",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
