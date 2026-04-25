from pathlib import Path

from src.config.config import DATA_DIR

def resolve_path(path, base_dir=DATA_DIR):
    path = Path(path)
    if path.is_absolute(): return path
    if path.exists(): return path.resolve()

    candidate = base_dir / path
    if candidate.exists(): return candidate.resolve()

    raise FileNotFoundError(f"File not found: {path}")

def list_files(data_dir=DATA_DIR, extension="csv"):
    return [f.relative_to(data_dir) for f in list(data_dir.rglob(f"*.{extension}"))]