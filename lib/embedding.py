"""Embedding model loading and encoding.

The single most important property here is identity. Every vector in the index
was produced by one specific model at one specific revision, and a different
model produces vectors in a different space. Nothing errors when they are
mismatched, retrieval just quietly returns irrelevant passages, so the model
identity is stamped into the index manifest and checked before querying.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

from .config import ROOT, apply_offline_env, resolve


@dataclass(frozen=True)
class EmbeddingIdentity:
    """What the manifest records and the Mac verifies before serving."""

    model: str
    revision: str
    dimension: int

    def fingerprint(self) -> str:
        raw = f"{self.model}@{self.revision}/{self.dimension}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["fingerprint"] = self.fingerprint()
        return out

    def matches(self, other: "EmbeddingIdentity") -> bool:
        return (
            self.model == other.model
            and self.revision == other.revision
            and self.dimension == other.dimension
        )


class EmbeddingModel:
    def __init__(self, settings: dict[str, Any], *, device: str | None = None) -> None:
        cfg = settings["Embedding"]
        self.name: str = cfg["Model"]
        self.revision: str = cfg.get("Revision") or "main"
        self.expected_dim: int = int(cfg["Dimension"])
        self.batch_size: int = int(cfg.get("BatchSize", 32))
        self.query_prefix: str = cfg.get("QueryPrefix", "")
        self.local_dir: Path = resolve(cfg.get("LocalDir", "models/embedding"))
        self._device = device or cfg.get("Device", "cpu")
        self._model = None

    def _resolve_device(self) -> str:
        if self._device != "cuda":
            return self._device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # Prefer the copied local snapshot. On the Mac that folder is the
            # only source, since the bundle runs with the hub disabled.
            if self.local_dir.is_dir() and any(self.local_dir.iterdir()):
                apply_offline_env()
                source = str(self.local_dir)
            else:
                source = self.name

            self._model = SentenceTransformer(
                source,
                device=self._resolve_device(),
                revision=None if source != self.name else self.revision,
            )
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def identity(self) -> EmbeddingIdentity:
        return EmbeddingIdentity(self.name, self.revision, self.dimension)

    def verify_dimension(self) -> None:
        actual = self.dimension
        if actual != self.expected_dim:
            raise RuntimeError(
                f"Embedding dimension mismatch: settings say {self.expected_dim}, "
                f"model {self.name} produced {actual}. Fix settings.json or the "
                f"index will be unusable."
            )

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def encode_query(self, text: str) -> list[float]:
        # bge-style models expect an instruction prefix on the query side only;
        # adding it to documents would shift them out of the indexed space.
        payload = f"{self.query_prefix}{text}" if self.query_prefix else text
        vector = self.model.encode(
            [payload],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        return vector.tolist()

    def save_snapshot(self, destination: Path | None = None) -> Path:
        """Persist the model locally so the Mac never needs the hub."""
        destination = destination or self.local_dir
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save(str(destination))
        (destination / "identity.json").write_text(
            json.dumps(self.identity().to_dict(), indent=2), encoding="utf-8"
        )
        return destination


def load_identity_file(path: Path) -> EmbeddingIdentity | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return EmbeddingIdentity(
        model=raw["model"], revision=raw["revision"], dimension=int(raw["dimension"])
    )
