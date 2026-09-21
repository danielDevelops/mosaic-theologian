"""Chunk, embed, and write everything into the three LanceDB collections.

Re-indexing a source deletes its existing rows first, so running this twice
updates in place rather than accumulating duplicates.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from lib import chunking
from lib.chunking import Chunk
from lib.embedding import EmbeddingModel
from lib.store import VectorStore
from workers.common import WorkerContext, base_parser, log


def embed_and_store(store: VectorStore, model: EmbeddingModel,
                    chunks: list[Chunk], collection: str) -> int:
    if not chunks:
        return 0
    vectors = model.encode_documents([c.text for c in chunks])
    return store.add(collection, chunks, vectors)


def index_bible(ctx: WorkerContext, store: VectorStore,
                model: EmbeddingModel) -> int:
    verses_path = ctx.paths.bible / "verses.jsonl"
    if not verses_path.is_file():
        log("  no verses.jsonl; skipping scripture.")
        return 0

    if store.count("scripture") > 0 and not ctx.force:
        log(f"  scripture already indexed ({store.count('scripture'):,} rows).")
        return 0

    cfg = ctx.settings["Chunking"]
    translation = ctx.settings["Bible"].get("Translation", "ESV")

    chapters: dict[tuple[str, int], dict[int, str]] = defaultdict(dict)
    with open(verses_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            chapters[(row["book_id"], int(row["chapter"]))][int(row["verse"])] = row["text"]

    log(f"  chunking {len(chapters):,} chapters")

    total = 0
    batch: list[Chunk] = []

    for (osis, chapter), verses in sorted(chapters.items()):
        batch.extend(chunking.chunk_bible_chapter(
            osis=osis,
            chapter=chapter,
            verses=verses,
            translation=translation,
            window=int(cfg.get("VerseWindowSize", 6)),
            stride=int(cfg.get("VerseWindowStride", 3)),
        ))

        # Flush periodically to keep peak memory bounded on a long book.
        if len(batch) >= 512:
            total += embed_and_store(store, model, batch, "scripture")
            batch = []
            print(f"    {total:,} scripture chunks", end="\r", flush=True)

    total += embed_and_store(store, model, batch, "scripture")
    log(f"  scripture: {total:,} chunks indexed")
    return total


def index_content(ctx: WorkerContext, store: VectorStore,
                  model: EmbeddingModel) -> int:
    cfg = ctx.settings["Chunking"]

    candidates = [
        job for job in ctx.jobs
        if job.at_least("transcribed") and (ctx.force or not job.at_least("indexed"))
    ]

    if not candidates:
        log("  no new content to index.")
        return 0

    log(f"  {len(candidates)} item(s) to index")
    total = 0
    processed = 0

    for job in candidates:
        if ctx.deadline.expired():
            log("  deadline reached; stopping cleanly.")
            break

        chunks: list[Chunk] = []
        collection = "beliefs" if job.kind == "belief" else "mosaic"

        transcript_path = (ctx.root / job.transcript_path) if job.transcript_path else None
        page_path = (ctx.root / job.page_path) if job.page_path else None

        if transcript_path and transcript_path.is_file():
            payload = json.loads(transcript_path.read_text(encoding="utf-8"))
            text = (payload.get("text") or "").strip()
            if text:
                chunks.extend(chunking.chunk_sermon(
                    source_id=job.id,
                    text=text,
                    title=job.title or payload.get("title", ""),
                    url=job.url,
                    date=job.date,
                    speaker=job.speaker,
                    series=job.series,
                    campus=job.campus,
                    target_tokens=int(cfg.get("SermonTargetTokens", 1000)),
                    overlap_tokens=int(cfg.get("SermonOverlapTokens", 150)),
                ))

        if page_path and page_path.is_file():
            page = json.loads(page_path.read_text(encoding="utf-8"))
            sections = [tuple(s) for s in page.get("sections", [])]
            if sections:
                if collection == "beliefs":
                    chunks.extend(chunking.chunk_beliefs(
                        source_id=f"{job.id}:page",
                        sections=sections,
                        title=job.title or page.get("title", ""),
                        url=job.url,
                    ))
                elif not chunks:
                    # Only index page prose when there is no transcript; the
                    # summary otherwise duplicates what the sermon already says.
                    chunks.extend(chunking.chunk_page(
                        source_id=f"{job.id}:page",
                        sections=sections,
                        title=job.title or page.get("title", ""),
                        url=job.url,
                        collection=collection,
                        max_tokens=int(cfg.get("PageMaxTokens", 900)),
                    ))

        if not chunks:
            job.advance("indexed")
            job.status = "done"
            job.error = "nothing indexable"
            ctx.jobs.put(job)
            continue

        # Replace rather than append, so a re-index cannot duplicate rows.
        store.delete_source(collection, job.id)
        store.delete_source(collection, f"{job.id}:page")

        added = embed_and_store(store, model, chunks, collection)
        total += added
        processed += 1

        job.advance("indexed")
        job.status = "done"
        job.error = ""
        ctx.jobs.put(job)

        log(f"    {job.id}: {added} chunks -> {collection}")

    log(f"  content: {total:,} chunks from {processed} item(s)")
    return total


def main() -> int:
    parser = base_parser("Build the vector index")
    args = parser.parse_args()

    with WorkerContext(args) as ctx:
        model = EmbeddingModel(ctx.settings)
        model.verify_dimension()
        identity = model.identity()
        log(f"Embedding model: {identity.model}@{identity.revision} "
            f"dim={identity.dimension}")

        store = VectorStore(ctx.paths.index, identity.dimension)

        # A pre-existing index built by a different model cannot be appended
        # to; the vectors would not share a space.
        if not ctx.force:
            store.assert_embedding_matches(identity)

        log("Indexing scripture")
        index_bible(ctx, store, model)

        log("Indexing Mosaic content")
        index_content(ctx, store, model)

        counts = store.counts()
        store.write_manifest(identity, extra={
            "translation": ctx.settings["Bible"].get("Translation", ""),
        })

        log("Index counts: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
