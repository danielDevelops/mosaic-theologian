"""Pin one CUDA toolkit on PATH before faster-whisper loads.

Windows will load the first cublas/cudnn it finds. Two toolkits (or a
toolkit plus a leftover CUDA 11 bin) on PATH is the usual "DLL not found"
failure. This module reads .env, keeps only the pinned version, and also
adds the venv nvidia-* wheel bins that CTranslate2 actually needs.
"""

from __future__ import annotations

import os
from pathlib import Path

_APPLIED = False


def _project_root() -> Path:
    override = os.environ.get("MOSAIC_ROOT")
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "config" / "settings.json").is_file():
            return candidate
    return here.parents[1]


def load_dotenv(root: Path | None = None) -> dict[str, str]:
    path = (root or _project_root()) / ".env"
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if not name:
            continue
        loaded[name] = value
        os.environ.setdefault(name, value)
    return loaded


def _toolkit_for(version: str) -> Path | None:
    base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if version:
        candidate = base / f"v{version}"
        if (candidate / "bin").is_dir():
            return candidate
    if not base.is_dir():
        return None
    versions = sorted(
        (p for p in base.iterdir() if p.is_dir() and p.name.startswith("v")),
        key=lambda p: p.name,
        reverse=True,
    )
    for folder in versions:
        if folder.name.startswith("v12") and (folder / "bin").is_dir():
            return folder
    return versions[0] if versions else None


def _venv_nvidia_bins(root: Path) -> list[Path]:
    nvidia = root / ".venv" / "Lib" / "site-packages" / "nvidia"
    if not nvidia.is_dir():
        return []
    bins: list[Path] = []
    for pkg in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
        folder = nvidia / pkg / "bin"
        if folder.is_dir():
            bins.append(folder)
    return bins


def _add_dll_dir(path: Path) -> None:
    adder = getattr(os, "add_dll_directory", None)
    if adder is None:
        return
    try:
        adder(str(path))
    except OSError:
        pass


def apply_cuda_env(root: Path | None = None) -> dict[str, str]:
    """Pin PATH to one CUDA version. Safe to call more than once."""
    global _APPLIED
    root = root or _project_root()
    load_dotenv(root)

    version = (os.environ.get("CUDA_VERSION") or "12.4").strip()
    raw_path = (os.environ.get("CUDA_PATH") or "").strip()
    toolkit = Path(raw_path) if raw_path else None
    if toolkit is None or not (toolkit / "bin").is_dir():
        toolkit = _toolkit_for(version)
    if toolkit is not None:
        os.environ["CUDA_PATH"] = str(toolkit)
        tag = toolkit.name.replace(".", "_").upper()
        os.environ.setdefault(f"CUDA_PATH_{tag}", str(toolkit))

    current = os.environ.get("PATH", "").split(os.pathsep)
    kept = [
        part for part in current
        if part and "NVIDIA GPU Computing Toolkit\\CUDA" not in part
        and "NVIDIA GPU Computing Toolkit/CUDA" not in part
    ]
    prepend: list[str] = []
    if toolkit is not None and (toolkit / "bin").is_dir():
        prepend.append(str(toolkit / "bin"))
        _add_dll_dir(toolkit / "bin")
    for folder in _venv_nvidia_bins(root):
        prepend.append(str(folder))
        _add_dll_dir(folder)

    os.environ["PATH"] = os.pathsep.join(prepend + kept)
    _APPLIED = True
    return {
        "cuda_version": version,
        "cuda_path": str(toolkit) if toolkit else "",
        "prepended": prepend,
    }
