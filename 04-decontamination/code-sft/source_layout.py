"""Dataset-card source scopes; auxiliary JSON is not automatically training data."""

import fnmatch
from pathlib import PurePosixPath

AUXILIARY_FILES = {
    "dataset_info.json",
    "dataset_infos.json",
    "state.json",
    "manifest.json",
    "tools.json",
}


def declared_data_patterns(card):
    patterns = []
    configs = card.get("configs", [])
    if isinstance(configs, list):
        for config in configs:
            specs = config.get("data_files", []) if isinstance(config, dict) else []
            if isinstance(specs, list):
                for spec in specs:
                    if not isinstance(spec, dict):
                        continue
                    values = spec.get("path", [])
                    patterns.extend(
                        [values]
                        if isinstance(values, str)
                        else values
                        if isinstance(values, list)
                        else []
                    )
    return [p for p in patterns if isinstance(p, str)]


def is_training_source_file(path, card):
    patterns = declared_data_patterns(card)
    # Preserve released data variants even when the card only advertises some
    # of them. Exclude known auxiliary files unless explicitly named as data.
    if PurePosixPath(path).name in AUXILIARY_FILES:
        return path in patterns
    return True
