import json
import hashlib

from src.settings.config import *

def make_signature(config):
    s = json.dumps(config, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]

def ensure(obj, cls):
    if isinstance(obj, cls): return obj
    return cls.from_name(obj)