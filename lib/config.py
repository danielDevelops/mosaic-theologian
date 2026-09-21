"""Settings loading and path resolution.

The project root is the folder containing config/settings.json. Every other
path in the project is resolved relative to it, so the whole tree can be moved
or copied to another machine without editing anything.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def find_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__).resolve()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "config" / "settings.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate config/settings.json in any parent directory."
    )


ROOT = find_root()


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_settings(root: Path | None = None) -> dict[str, Any]:
    root = root or ROOT
    with open(root / "config" / "settings.json", encoding="utf-8") as handle:
        settings = json.load(handle)

    # settings.local.json is an optional per-machine override. The Mac uses it
    # to switch the embedding device to cpu without touching the shared file.
    local = root / "config" / "settings.local.json"
    if local.is_file():
        with open(local, encoding="utf-8") as handle:
            settings = _deep_merge(settings, json.load(handle))

    return settings


def resolve(relative: str, root: Path | None = None) -> Path:
    """Turn a settings-relative path into an absolute one."""
    root = root or ROOT
    path = Path(relative)
    return path if path.is_absolute() else (root / path)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class Paths:
    """Canonical locations, so no module hardcodes a directory name."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or ROOT
        self.config = self.root / "config"
        self.state = self.root / "state"
        self.data = self.root / "data"
        self.bible = self.data / "bible"
        self.pages = self.data / "pages"
        self.audio = self.data / "audio"
        self.transcripts = self.data / "transcripts"
        self.inbox = self.data / "inbox"
        self.index = self.root / "index"
        self.models = self.root / "models"

    def ensure_all(self) -> None:
        for path in (
            self.state,
            self.bible,
            self.pages,
            self.audio,
            self.transcripts,
            self.inbox,
            self.index,
            self.models,
        ):
            ensure_dir(path)


def offline_env() -> dict[str, str]:
    """Environment flags that stop the embedding library phoning home.

    Without these, sentence-transformers contacts the model hub on load and
    stalls or fails outright on a machine intended to run offline.
    """
    return {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }


def apply_offline_env() -> None:
    os.environ.update(offline_env())
