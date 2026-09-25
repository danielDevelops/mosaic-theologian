"""LanceDB vector store.

Three collections so retrieval can be weighted by source authority:
scripture, beliefs, mosaic. LanceDB is a plain folder on disk, which is what
makes the Windows-to-Mac handoff a file copy rather than a database migration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .chunking import Chunk
from .embedding import EmbeddingIdentity

COLLECTIONS = ("scripture", "beliefs", "mosaic")
MANIFEST_NAME = "index-manifest.json"

_INDEX_READ_FAILED = (
    "Index read failed for {collection}. LanceDB on this machine does not "
    "match the one that built the index."
)


class IndexReadError(RuntimeError):
    """The index could not be read. An empty result means something else."""


def schema_for(dimension: int):
    import pyarrow as pa

    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("collection", pa.string()),
        pa.field("text", pa.string()),
        pa.field("title", pa.string()),
        pa.field("citation", pa.string()),
        pa.field("url", pa.string()),
        pa.field("date", pa.string()),
        pa.field("speaker", pa.string()),
        pa.field("series", pa.string()),
        pa.field("campus", pa.string()),
        pa.field("book_id", pa.string()),
        pa.field("chapter", pa.int32()),
        pa.field("verse_start", pa.int32()),
        pa.field("verse_end", pa.int32()),
        pa.field("scripture_refs", pa.string()),
        pa.field("primary_refs", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dimension)),
    ])


class VectorStore:
    def __init__(self, index_dir: Path, dimension: int) -> None:
        self.index_dir = index_dir
        self.dimension = dimension
        self.index_dir.mkdir(parents=True, exist_ok=True)
        import lancedb

        self._db = lancedb.connect(str(self.index_dir))

    def _table(self, collection: str, create: bool = True):
        names = set(self._db.table_names())
        if collection in names:
            return self._db.open_table(collection)
        if not create:
            return None
        return self._db.create_table(
            collection, schema=schema_for(self.dimension), mode="create"
        )

    @staticmethod
    def _ensure_primary_refs(table) -> None:
        """Add the column on an index built before sermon primary refs existed.

        A full Reindex rewrites the tables. This keeps the nightly index of
        one new sermon from failing on the old schema in the meantime.
        """
        try:
            names = set(table.schema.names)
        except Exception:
            return
        if "primary_refs" in names:
            return
        try:
            table.add_columns({"primary_refs": "''"})
        except Exception:
            return

    def close(self) -> None:
        """Drop the connection so the index directory can be renamed."""
        db = getattr(self, "_db", None)
        self._db = None
        if db is None:
            return
        closer = getattr(db, "close", None)
        if callable(closer):
            closer()
        import gc
        gc.collect()

    def add(self, collection: str, chunks: Sequence[Chunk],
            vectors: Sequence[Sequence[float]]) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError("chunk/vector count mismatch")

        rows: list[dict[str, Any]] = []
        for chunk, vector in zip(chunks, vectors):
            row = chunk.to_row()
            row["vector"] = list(vector)
            rows.append(row)

        table = self._table(collection)
        self._ensure_primary_refs(table)
        table.add(rows)
        return len(rows)

    def delete_source(self, collection: str, source_id: str) -> None:
        """Drop every chunk from one source so a re-index cannot duplicate it."""
        table = self._table(collection, create=False)
        if table is None:
            return
        escaped = source_id.replace("'", "''")
        table.delete(f"source_id = '{escaped}'")

    def has_source(self, collection: str, source_id: str) -> bool:
        table = self._table(collection, create=False)
        if table is None:
            return False
        escaped = source_id.replace("'", "''")
        try:
            got = table.search().where(f"source_id = '{escaped}'").limit(1).to_list()
            return bool(got)
        except Exception:
            return False

    def search(self, collection: str, vector: Sequence[float],
               limit: int) -> list[dict[str, Any]]:
        table = self._table(collection, create=False)
        if table is None:
            return []
        try:
            results = table.search(list(vector)).limit(limit).to_list()
        except Exception as exc:
            raise IndexReadError(_INDEX_READ_FAILED.format(collection=collection)) from exc

        for row in results:
            row.pop("vector", None)
            # LanceDB returns L2 distance on normalized vectors; map it to a
            # 0..1 similarity so scores are comparable across collections.
            distance = float(row.pop("_distance", 0.0))
            row["score"] = max(0.0, 1.0 - distance / 2.0)
            row["collection"] = collection
        return results

    def filter_rows(self, collection: str, where: str,
                    columns: list[str] | None = None,
                    limit: int | None = 500) -> list[dict[str, Any]]:
        """Metadata lookup. Used to count a sermon's refs and to fetch a passage.

        `limit` None keeps every match. The related-sermon scan needs the whole
        library; a 500-row cap would drop sermons past the first page.
        """
        table = self._table(collection, create=False)
        if table is None:
            return []
        rows: list[dict[str, Any]] = []
        try:
            scanner = table.to_lance().scanner(filter=where, columns=columns)
            rows = scanner.to_table().to_pylist()
        except Exception:
            try:
                query = table.search().where(where)
                if limit is not None:
                    query = query.limit(limit)
                rows = query.to_list()
            except Exception as exc:
                raise IndexReadError(
                    _INDEX_READ_FAILED.format(collection=collection)
                ) from exc
        if limit is not None and len(rows) > limit:
            rows = rows[:limit]
        for row in rows:
            row.pop("vector", None)
        return rows

    def count(self, collection: str) -> int:
        table = self._table(collection, create=False)
        if table is None:
            return 0
        try:
            return int(table.count_rows())
        except Exception:
            return 0

    def counts(self) -> dict[str, int]:
        return {name: self.count(name) for name in COLLECTIONS}

    def write_manifest(self, identity: EmbeddingIdentity,
                       extra: dict[str, Any] | None = None) -> Path:
        payload = {
            "embedding": identity.to_dict(),
            "counts": self.counts(),
        }
        if extra:
            payload.update(extra)
        path = self.index_dir / MANIFEST_NAME
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def read_manifest(self) -> dict[str, Any] | None:
        path = self.index_dir / MANIFEST_NAME
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def assert_embedding_matches(self, identity: EmbeddingIdentity) -> None:
        """Refuse to serve if the query model differs from the index model.

        This is the failure that produces no error and no obvious symptom, only
        steadily irrelevant answers, so it is checked rather than assumed.
        """
        manifest = self.read_manifest()
        if not manifest:
            return
        recorded = manifest.get("embedding") or {}
        stored = EmbeddingIdentity(
            model=recorded.get("model", ""),
            revision=recorded.get("revision", ""),
            dimension=int(recorded.get("dimension", 0)),
        )
        if not stored.model:
            return
        if not stored.matches(identity):
            raise RuntimeError(
                "Embedding model mismatch between this machine and the index.\n"
                f"  index was built with: {stored.model}@{stored.revision} "
                f"dim={stored.dimension}\n"
                f"  this machine loaded:  {identity.model}@{identity.revision} "
                f"dim={identity.dimension}\n"
                "Retrieval would silently return irrelevant passages. Copy the "
                "bundled embedding model, or rebuild the index."
            )
