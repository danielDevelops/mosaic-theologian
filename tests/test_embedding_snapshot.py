"""A 6.1 embedding snapshot must load on sentence-transformers 5.7.

The bundled modules.json is checksummed. The loader rewrites a copy.

    python tests/test_embedding_snapshot.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.embedding import (  # noqa: E402
    _ST5_NORMALIZE,
    _ST6_NORMALIZE,
    rewritten_modules,
    snapshot_load_dir,
)


def check(name: str, ok: bool) -> bool:
    print(("ok  " if ok else "FAIL") + "  " + name)
    return ok


def main() -> int:
    original = [
        {"idx": 0, "type": "sentence_transformers.base.modules.transformer.Transformer"},
        {"idx": 2, "type": _ST6_NORMALIZE},
    ]
    kept = rewritten_modules(original, normalize_importable=True)
    ok = check("6.1 class stays when it imports", kept is None)
    ok &= check(
        "caller list is unchanged",
        original[1]["type"] == _ST6_NORMALIZE,
    )

    updated = rewritten_modules(original, normalize_importable=False)
    ok &= check(
        "5.7 path is substituted",
        updated is not None and updated[1]["type"] == _ST5_NORMALIZE,
    )
    ok &= check(
        "other modules are kept",
        updated is not None and updated[0]["type"].endswith("Transformer"),
    )
    ok &= check(
        "unrelated snapshot is left alone",
        rewritten_modules([{"type": "sentence_transformers.models.Normalize"}],
                          normalize_importable=False) is None,
    )

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "embedding"
        source.mkdir()
        (source / "model.safetensors").write_bytes(b"weights")
        (source / "modules.json").write_text(
            json.dumps(original, indent=2) + "\n", encoding="utf-8"
        )
        bundled = (source / "modules.json").read_text(encoding="utf-8")
        from lib import embedding as embedding_mod
        real = embedding_mod.class_is_importable
        embedding_mod.class_is_importable = lambda _name: False
        try:
            loaded = snapshot_load_dir(source)
        finally:
            embedding_mod.class_is_importable = real

        copied = json.loads((loaded / "modules.json").read_text(encoding="utf-8"))
        ok &= check("bundled modules.json is untouched",
                    (source / "modules.json").read_text(encoding="utf-8") == bundled)
        ok &= check("load dir is not the bundle", loaded != source)
        ok &= check("load dir uses the 5.7 class",
                    copied[1]["type"] == _ST5_NORMALIZE)
        ok &= check("weights are reachable",
                    (loaded / "model.safetensors").read_bytes() == b"weights")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
