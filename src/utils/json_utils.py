import json

from src.settings.config import *

def save_json(data, path):
    with open(path, "w") as f: json.dump(data, f, indent=4, default=str)