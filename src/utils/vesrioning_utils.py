import re
import hashlib
import json

def make_signature(config):
    s = json.dumps(config, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]

def get_versions(base_name, directory, extension="csv"):
    existing_files = list(directory.glob(f"{base_name}_v*.{extension}"))

    versions = []
    for f in existing_files:
        match = re.search(r"_v(\d+)", f.stem)
        if match: versions.append(int(match.group(1)))

    return versions

def get_latest_version(base_name, directory, extension="csv"):
    return max(get_versions(base_name, directory, extension), default=0)

def get_next_version(base_name, directory, extension="csv"):
    return get_latest_version(base_name, directory, extension) + 1