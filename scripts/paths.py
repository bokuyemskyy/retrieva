from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def data_path(*parts):
    return PROJECT_ROOT / "evals" / Path(*parts)
