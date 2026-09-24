"""Assemble the portable bundle the Mac consumes.

Two modes:

  --full        first handoff, including the model files (roughly 5-10 GB)
  --index-only  later re-syncs; the models never change when a sermon is
                added, so only the index and content travel (tens to a few
                hundred MB)

Every file is checksummed into manifest.json. The Mac replays those checksums
before serving, which is what catches a copy that silently truncated.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from lib.chat_model import ensure_chat_model
from lib.config import Paths, load_settings
from lib.embedding import EmbeddingModel
from lib.store import VectorStore
from workers.common import base_parser, log

CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def copy_tree(source: Path, target: Path) -> list[Path]:
    """Copy a directory and return the files written."""
    if not source.exists():
        return []
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("*.part"))
    return [p for p in target.rglob("*") if p.is_file()]


def human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def main() -> int:
    parser = base_parser("Export the portable bundle")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--full", action="store_true",
                        help="Include model files")
    parser.add_argument("--index-only", action="store_true",
                        help="Index and content only; models unchanged")
    args = parser.parse_args()

    settings = load_settings()
    paths = Paths()
    destination = Path(args.destination)
    include_models = bool(args.full) or not args.index_only

    manifest_path = paths.index / "index-manifest.json"
    if not manifest_path.is_file():
        log("No index to export. Run the night job first.")
        return 1

    index_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embedding_identity = index_manifest.get("embedding", {})

    if include_models:
        try:
            ensure_chat_model(settings, paths.root)
        except Exception as exc:
            log(f"ERROR: {exc}")
            log("ERROR: Export stopped. The bundle was not copied.")
            return 1

    destination.mkdir(parents=True, exist_ok=True)
    log(f"Destination: {destination}")

    written: list[Path] = []

    log("  index/")
    written += copy_tree(paths.index, destination / "index")

    if settings["Export"].get("IncludeTranscripts", True):
        log("  data/transcripts/")
        written += copy_tree(paths.transcripts, destination / "data" / "transcripts")

    if settings["Export"].get("IncludePages", True):
        log("  data/pages/")
        written += copy_tree(paths.pages, destination / "data" / "pages")

    # Scripture metadata travels; the raw source file does not need to.
    bible_meta = paths.bible / "bible-meta.json"
    if bible_meta.is_file():
        target = destination / "data" / "bible" / "bible-meta.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bible_meta, target)
        written.append(target)

    config_target = destination / "config"
    config_target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths.config / "settings.json", config_target / "settings.json")
    written.append(config_target / "settings.json")

    # The Mac has no CUDA and no GPU path, so pin the embedding device there.
    local_override = {"Embedding": {"Device": "cpu"}}
    (config_target / "settings.local.json").write_text(
        json.dumps(local_override, indent=2), encoding="utf-8"
    )
    written.append(config_target / "settings.local.json")

    if include_models:
        log("  models/  (this is the large part)")
        embedding_dir = paths.root / settings["Embedding"].get(
            "LocalDir", "models/embedding")
        if embedding_dir.is_dir():
            written += copy_tree(embedding_dir, destination / "models" / "embedding")
        else:
            log("    WARNING: embedding model not cached locally.")
            log("    Run EnsureDeps first, or the Mac will have to fetch it and "
                "may get a different revision.")

        chat_model = paths.root / settings["Chat"]["ModelFile"]
        target = destination / "models" / chat_model.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(chat_model, target)
        written.append(target)
    else:
        log("  models/ skipped (index-only). Models do not change when "
            "sermons are added.")

    # Ship the runtime so the Mac bundle is self-contained.
    for folder in ("lib", "query"):
        written += copy_tree(paths.root / folder, destination / folder)

    for name in ("mosaic.sh", "requirements-mac.txt"):
        source = paths.root / name
        if source.is_file():
            shutil.copy2(source, destination / name)
            written.append(destination / name)

    docs_source = paths.root / "docs" / "README-Mac.md"
    if docs_source.is_file():
        (destination / "docs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(docs_source, destination / "docs" / "README-Mac.md")
        written.append(destination / "docs" / "README-Mac.md")

    log("  hashing...")
    files: dict[str, dict] = {}
    total_bytes = 0

    for path in sorted(set(written)):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = str(path.relative_to(destination)).replace("\\", "/")
        size = path.stat().st_size
        total_bytes += size
        files[relative] = {"sha256": sha256_of(path), "bytes": size}

    store = VectorStore(destination / "index",
                        int(embedding_identity.get("dimension", 384)))
    counts = store.counts()

    manifest = {
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "full" if include_models else "index-only",
        "embedding": embedding_identity,
        "counts": counts,
        "translation": index_manifest.get("translation", ""),
        "totalBytes": total_bytes,
        "fileCount": len(files),
        "includesModels": include_models,
        "files": files,
    }

    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    log("")
    log(f"Bundle written: {len(files)} files, {human(total_bytes)}")
    log("Index rows: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))
    log(f"Embedding:  {embedding_identity.get('model')}@"
        f"{embedding_identity.get('revision')} "
        f"dim={embedding_identity.get('dimension')}")

    if not include_models:
        log("")
        log("Index-only bundle. The Mac keeps the models it already has, and "
            "verify still checks they match this index.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
