import json
import pandas as pd
from pathlib import Path

from src.settings.config import DATA_DIR

def resolve_path(path, base_dir=DATA_DIR):
    path = Path(path)
    if path.is_absolute(): return path

    if path.exists(): return path.resolve()

    candidate = Path(base_dir) / path #careful here
    if candidate.exists(): return candidate.resolve()

    raise FileNotFoundError(f"File not found: {path} (Checked absolute and base_dir: {base_dir})")

def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f: 
        json.dump(data, f, indent=4, default=str)

def load_json(path, base_dir=None):
    final_path = resolve_path(path, base_dir) if base_dir else Path(path)
    with open(final_path, "r") as f: 
        return json.load(f)

def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path

def load_csv(path, base_dir=DATA_DIR):
    return pd.read_csv(resolve_path(path, base_dir), index_col=0, parse_dates=True)

def list_contents(directory, pattern="*"):
    path = Path(directory)
    if not path.exists(): return []
    return [f.name for f in path.glob(pattern)]