import json
import pandas as pd
from pathlib import Path
import shutil
import os

from src.settings.config import *

from src.utils.logging_utils import get_logger

logger = get_logger("utils.io_utils")

def resolve_path(path, base_dir=DATA_DIR):
    path = Path(path)
    if path.is_absolute(): return path

    if path.exists(): return path.resolve()

    candidate = Path(base_dir) / path
    if candidate.exists(): return candidate.resolve()

    logger.error(f"File not found: {path} (Checked absolute and base_dir: {base_dir})")

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

def delete_file(path, base_dir=DATA_DIR):
    target = resolve_path(path, base_dir)
    if target.is_file():
        os.remove(target)

        logger.info(f"File {target} has been deleted.")
    else:
        logger.error(f"Target {target} is not a file. Use delete_directory instead.")

        raise ValueError(f"Target {target} is not a file. Use delete_directory instead.")

def delete_directory(path, base_dir=MODELS_DIR):
    target = resolve_path(path, base_dir)
    if target.is_dir():
        shutil.rmtree(target)

        logger.info(f"Directory {target} has been deleted.")
    else:
        logger.error(f"Target {target} is not a directory. Use delete_file instead.")

        raise ValueError(f"Target {target} is not a directory. Use delete_file instead.")