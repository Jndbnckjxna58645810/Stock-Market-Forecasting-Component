import pandas as pd
from pathlib import Path
import re

from src.config.config import *

def get_versions(base_name, directory):
    existing_files = list(directory.glob(f"{base_name}_v*.csv"))

    versions = []
    for f in existing_files:
        match = re.search(r"_v(\d+)", f.stem)
        if match: versions.append(int(match.group(1)))

    return versions

def get_next_version(base_name, directory): return max(get_versions(base_name, directory), default=0) + 1

def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)

def save_processed_csv(df, ticker, start_date, end_date, interval="1d", path=None):
    base_name = f"{ticker}_{start_date}_{end_date}_{interval}"
    save_csv(df, PROCESSED_DATA_DIR / f"{base_name}_v{get_next_version(base_name, PROCESSED_DATA_DIR)}.csv" if path == None else path)
    return path

def resolve_path(path, base_dir=DATA_DIR):
    path = Path(path)
    if path.is_absolute(): return path
    if path.exists(): return path.resolve()

    candidate = base_dir / path
    if candidate.exists(): return candidate.resolve()

    raise FileNotFoundError(f"File not found: {path}")

def load_csv(path): return pd.read_csv(resolve_path(path), index_col=0, parse_dates=True)

def list_datasets(data_dir=DATA_DIR): return [f.relative_to(data_dir) for f in list(data_dir.rglob("*.csv"))]