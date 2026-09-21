"""Chunking for each source type.

What gets stored is source text, never questions. Each chunk carries enough
metadata to rebuild a citation without another lookup.

Token counts are approximated from whitespace words rather than by loading a
tokenizer. The chunker only needs to hit a size band, and an exact count buys
nothing here while costing a dependency on both machines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

from . import bookmap

WORDS_TO_TOKENS = 1.33


def count_tokens(text: str) -> int:
    return int(len(text.split()) * WORDS_TO_TOKENS)


@dataclass
class Chunk:
    """One indexed passage. `collection` selects scripture/beliefs/mosaic."""

    id: str
    collection: str
    text: str
    title: str = ""
    citation: str = ""
    url: str = ""
    date: str = ""
    speaker: str = ""
    series: str = ""
    campus: str = ""
    book_id: str = ""
    chapter: int = 0
    verse_start: int = 0
    verse_end: int = 0
    scripture_refs: list[str] = field(default_factory=list)
    source_id: str = ""

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["scripture_refs"] = " ".join(self.scripture_refs)
        return row


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_sermon(
    *,
    source_id: str,
    text: str,
    title: str,
    url: str,
    date: str = "",
    speaker: str = "",
    series: str = "",
    campus: str = "",
    target_tokens: int = 1000,
    overlap_tokens: int = 150,
) -> list[Chunk]:
    """Sentence-aligned windows of roughly `target_tokens` with overlap.

    Splitting on sentence boundaries rather than a raw token offset keeps
    quotations intact, which matters because these chunks get read back to the
    model verbatim.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    window: list[str] = []
    window_tokens = 0
    index = 0

    def flush() -> None:
        nonlocal window, window_tokens, index
        if not window:
            return
        body = " ".join(window).strip()
        if not body:
            return
        chunks.append(Chunk(
            id=f"{source_id}#c{index:04d}",
            collection="mosaic",
            text=body,
            title=title,
            citation=_sermon_citation(title, speaker, date),
            url=url,
            date=date,
            speaker=speaker,
            series=series,
            campus=campus,
            scripture_refs=bookmap.extract_refs(body),
            source_id=source_id,
        ))
        index += 1

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if window and window_tokens + sentence_tokens > target_tokens:
            flush()
            # Carry the tail of the previous window forward as overlap.
            carry: list[str] = []
            carried = 0
            for prior in reversed(window):
                prior_tokens = count_tokens(prior)
                if carried + prior_tokens > overlap_tokens:
                    break
                carry.insert(0, prior)
                carried += prior_tokens
            window = carry
            window_tokens = carried

        window.append(sentence)
        window_tokens += sentence_tokens

    flush()
    return chunks


def _sermon_citation(title: str, speaker: str, date: str) -> str:
    bits = [b for b in (title, speaker, date) if b]
    return " | ".join(bits)


def chunk_page(
    *,
    source_id: str,
    sections: Iterable[tuple[str, str]],
    title: str,
    url: str,
    collection: str = "mosaic",
    max_tokens: int = 900,
) -> list[Chunk]:
    """One chunk per heading section, split further only if oversized."""
    chunks: list[Chunk] = []
    index = 0

    for heading, body in sections:
        body = (body or "").strip()
        if not body:
            continue

        pieces = [body]
        if count_tokens(body) > max_tokens:
            pieces = []
            current: list[str] = []
            current_tokens = 0
            for sentence in _split_sentences(body):
                tokens = count_tokens(sentence)
                if current and current_tokens + tokens > max_tokens:
                    pieces.append(" ".join(current))
                    current, current_tokens = [], 0
                current.append(sentence)
                current_tokens += tokens
            if current:
                pieces.append(" ".join(current))

        for piece in pieces:
            label = f"{title} — {heading}".strip(" —") if heading else title
            chunks.append(Chunk(
                id=f"{source_id}#s{index:04d}",
                collection=collection,
                text=(f"{heading}\n\n{piece}" if heading else piece),
                title=title,
                citation=label,
                url=url,
                scripture_refs=bookmap.extract_refs(piece),
                source_id=source_id,
            ))
            index += 1

    return chunks


def chunk_beliefs(
    *,
    source_id: str,
    sections: Iterable[tuple[str, str]],
    title: str,
    url: str,
) -> list[Chunk]:
    """One chunk per belief statement, kept whole.

    Belief statements are short and self-contained, and a partial doctrinal
    statement is worse than none, so these are never split.
    """
    chunks: list[Chunk] = []
    for index, (heading, body) in enumerate(sections):
        body = (body or "").strip()
        if not body:
            continue
        chunks.append(Chunk(
            id=f"{source_id}#b{index:04d}",
            collection="beliefs",
            text=(f"{heading}\n\n{body}" if heading else body),
            title=title,
            citation=f"{title} — {heading}".strip(" —") if heading else title,
            url=url,
            scripture_refs=bookmap.extract_refs(body),
            source_id=source_id,
        ))
    return chunks


def chunk_bible_chapter(
    *,
    osis: str,
    chapter: int,
    verses: dict[int, str],
    translation: str = "ESV",
    window: int = 6,
    stride: int = 3,
) -> list[Chunk]:
    """Whole chapter plus overlapping verse windows.

    The chapter chunk answers "what does this passage say" questions; the
    smaller windows give retrieval something tightly scoped to match against
    so a single verse is not buried inside a long chapter.
    """
    if not verses:
        return []

    ordered = sorted(verses.items())
    chunks: list[Chunk] = []
    book = bookmap.display_name(osis)

    chapter_text = " ".join(f"{num} {text}".strip() for num, text in ordered)
    first, last = ordered[0][0], ordered[-1][0]
    chunks.append(Chunk(
        id=f"bible:{osis}.{chapter}",
        collection="scripture",
        text=chapter_text,
        title=f"{book} {chapter}",
        citation=f"{book} {chapter} ({translation})",
        book_id=osis,
        chapter=chapter,
        verse_start=first,
        verse_end=last,
        source_id=f"bible:{osis}",
    ))

    if len(ordered) > window:
        for start in range(0, len(ordered), stride):
            slice_ = ordered[start:start + window]
            if len(slice_) < 2:
                break
            body = " ".join(f"{num} {text}".strip() for num, text in slice_)
            v_first, v_last = slice_[0][0], slice_[-1][0]
            chunks.append(Chunk(
                id=f"bible:{osis}.{chapter}.{v_first}-{v_last}",
                collection="scripture",
                text=body,
                title=f"{book} {chapter}:{v_first}-{v_last}",
                citation=f"{bookmap.format_ref(osis, chapter, v_first, v_last)} ({translation})",
                book_id=osis,
                chapter=chapter,
                verse_start=v_first,
                verse_end=v_last,
                source_id=f"bible:{osis}",
            ))
            if start + window >= len(ordered):
                break

    return chunks
