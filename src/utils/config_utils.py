import json

from src.settings.config import *
from src.utils.path_utils import resolve_path

def load_config(path, directory):
    with open(resolve_path(path, directory), "r") as f: return json.load(f)

def ensure(obj, cls):
    if isinstance(obj, cls): return obj
    return cls.from_name(obj)