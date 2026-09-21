"""Verify a copied bundle before serving from it.

Two classes of problem are caught here, and both are silent otherwise:

  1. A copy that truncated. Checksums catch it. The alternative is an index
     that looks healthy and quietly answers nothing for whatever was lost.
  2. An embedding model that differs from the one that built the index.
     Vectors would be in a different space, so retrieval would return
     irrelevant passages without raising anything.

Exit codes: 0 verified, 1 bundle problem, 2 embedding mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import Paths, apply_offline_env, load_settings  # noqa: E402
from lib.embedding import EmbeddingModel                        # noqa: E402
from lib.store import VectorStore                               # noqa: E402

CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the copied bundle")
    parser.add_argument("--quick", action="store_true",
                        help="Check sizes only, skip checksums")
    parser.add_argument("--skip-embedding", action="store_true",
                        help="Skip loading the embedding model")
    args = parser.parse_args()

    root = Paths().root
    manifest_path = root / "manifest.json"

    if not manifest_path.is_file():
        print("FAIL  manifest.json not found.")
        print("      This folder is not an exported bundle. Copy the whole")
        print("      mosaic-portable directory, not just the index.")
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    print(f"Bundle created {manifest.get('created')} "
          f"({manifest.get('mode')}, {len(files)} files)")

    missing: list[str] = []
    corrupt: list[str] = []
    checked = 0

    for relative, expected in files.items():
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue

        if path.stat().st_size != expected["bytes"]:
            corrupt.append(f"{relative} (size {path.stat().st_size} != "
                           f"{expected['bytes']})")
            continue

        if not args.quick:
            if sha256_of(path) != expected["sha256"]:
                corrupt.append(f"{relative} (checksum)")
                continue

        checked += 1
        if checked % 200 == 0:
            print(f"  {checked}/{len(files)} verified", end="\r", flush=True)

    print(f"  {checked}/{len(files)} files verified" + " " * 20)

    if missing:
        print(f"\nFAIL  {len(missing)} file(s) missing from the copy:")
        for name in missing[:10]:
            print(f"      {name}")
        if len(missing) > 10:
            print(f"      ... and {len(missing) - 10} more")

    if corrupt:
        print(f"\nFAIL  {len(corrupt)} file(s) do not match the manifest:")
        for name in corrupt[:10]:
            print(f"      {name}")
        if len(corrupt) > 10:
            print(f"      ... and {len(corrupt) - 10} more")

    if missing or corrupt:
        print("\n      The copy is incomplete. Re-copy the bundle rather than")
        print("      serving from it: retrieval would quietly return partial")
        print("      results with no error.")
        return 1

    expected_counts = manifest.get("counts", {})
    embedding_recorded = manifest.get("embedding", {})

    print("\nIndex rows recorded: " +
          ", ".join(f"{k}={v:,}" for k, v in expected_counts.items()))

    apply_offline_env()
    settings = load_settings()

    try:
        store = VectorStore(root / "index",
                            int(embedding_recorded.get("dimension", 384)))
        actual_counts = store.counts()
    except Exception as exc:
        print(f"FAIL  index unreadable: {exc}")
        return 1

    print("Index rows present:  " +
          ", ".join(f"{k}={v:,}" for k, v in actual_counts.items()))

    for collection, expected_rows in expected_counts.items():
        if actual_counts.get(collection, 0) != expected_rows:
            print(f"\nFAIL  {collection}: expected {expected_rows:,} rows, "
                  f"found {actual_counts.get(collection, 0):,}")
            return 1

    if actual_counts.get("scripture", 0) == 0:
        print("\nWARN  scripture collection is empty. Bible-first answers will")
        print("      not work until the index is rebuilt with a Bible source.")

    if args.skip_embedding:
        print("\nOK    bundle verified (embedding check skipped).")
        return 0

    print("\nLoading embedding model to confirm it matches the index...")
    try:
        model = EmbeddingModel(settings, device="cpu")
        identity = model.identity()
    except Exception as exc:
        print(f"FAIL  could not load the embedding model: {exc}")
        print("      The bundle should contain models/embedding. Re-export")
        print("      with -Full if it is missing.")
        return 2

    print(f"  index built with: {embedding_recorded.get('model')}@"
          f"{embedding_recorded.get('revision')} "
          f"dim={embedding_recorded.get('dimension')}")
    print(f"  this machine has: {identity.model}@{identity.revision} "
          f"dim={identity.dimension}")

    try:
        store.assert_embedding_matches(identity)
    except RuntimeError as exc:
        print(f"\nFAIL  {exc}")
        return 2

    print("\nOK    bundle verified. Start with: ./mosaic.sh start")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
