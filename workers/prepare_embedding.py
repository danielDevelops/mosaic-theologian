"""Download the embedding model once and snapshot it into models/embedding.

The Mac runs with the model hub disabled, so the exact files that built the
index have to travel in the bundle. Resolving the model by name on the Mac
would risk pulling a different revision, which produces vectors in a
different space and silently degrades every answer.
"""

from __future__ import annotations

from lib.config import Paths, load_settings
from lib.embedding import EmbeddingModel
from workers.common import base_parser, log


def main() -> int:
    parser = base_parser("Cache the embedding model locally")
    args = parser.parse_args()

    settings = load_settings()
    paths = Paths()
    paths.ensure_all()

    target = paths.root / settings["Embedding"].get("LocalDir", "models/embedding")

    if target.is_dir() and any(target.iterdir()) and not args.force:
        log(f"Embedding model already cached at {target}")
        return 0

    model = EmbeddingModel(settings)
    log(f"Fetching {model.name}@{model.revision}")

    try:
        model.verify_dimension()
    except RuntimeError as exc:
        log(f"ERROR: {exc}")
        return 2

    saved = model.save_snapshot(target)
    identity = model.identity()
    log(f"Cached to {saved}")
    log(f"Identity: {identity.model}@{identity.revision} "
        f"dim={identity.dimension} fingerprint={identity.fingerprint()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
