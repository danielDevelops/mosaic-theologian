"""Retrieval across the three collections.

The question is embedded at ask time. Nothing is pre-seeded: the index holds
source text only, so any question can be asked and the retriever finds
whatever is closest to it.

Collections are weighted rather than merged blindly, which is how the source
hierarchy (Scripture, then beliefs, then sermons) is expressed in scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import bookmap
from .embedding import EmbeddingModel
from .store import COLLECTIONS, VectorStore


@dataclass
class Passage:
    collection: str
    text: str
    citation: str
    url: str = ""
    title: str = ""
    date: str = ""
    speaker: str = ""
    score: float = 0.0
    raw_score: float = 0.0
    source_id: str = ""
    primary_refs: list[str] = field(default_factory=list)
    scripture_refs: list[str] = field(default_factory=list)
    connection: str = ""

    @property
    def label(self) -> str:
        if self.collection == "scripture":
            return self.citation
        bits = [b for b in (self.citation or self.title, self.date) if b]
        return " | ".join(dict.fromkeys(bits))


@dataclass
class RetrievalResult:
    question: str
    passages: list[Passage] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def by_collection(self, name: str) -> list[Passage]:
        return [p for p in self.passages if p.collection == name]

    @property
    def mosaic_is_silent(self) -> bool:
        """True when nothing from Mosaic's own material is relevant.

        The prompt uses this to tell the model to say so explicitly instead of
        inventing a position the church never took.
        """
        relevant = [p for p in self.passages
                    if p.collection in ("mosaic", "beliefs") and p.raw_score > 0.35]
        return not relevant


def ref_boost_amount(collection: str, citation: str, primary_packed: str,
                     mentioned_packed: str, asked_refs: set[str],
                     primary_boost: float, mentioned_boost: float) -> float:
    """Extra score when the question names a passage this row is about or cites."""
    if not asked_refs:
        return 0.0
    if collection == "scripture":
        if bookmap.any_overlap(asked_refs, bookmap.extract_refs(citation)):
            return primary_boost
        return 0.0

    primary = bookmap.unpack_refs(primary_packed)
    mentioned = bookmap.unpack_refs(mentioned_packed)
    if bookmap.any_overlap(asked_refs, primary):
        return primary_boost
    if bookmap.any_overlap(asked_refs, mentioned):
        return mentioned_boost
    # Older rows have no primary_refs. A title that names the passage still counts.
    if bookmap.any_overlap(asked_refs, bookmap.extract_refs(citation)):
        return primary_boost
    return 0.0


def _ref_span(ref: str) -> int:
    parsed = bookmap.parse_ref(ref)
    if not parsed or parsed[2] is None:
        return 10_000
    end = parsed[3] if parsed[3] is not None else parsed[2]
    return end - parsed[2]


def sustained_mentions(per_chunk: list[list[str]], primary: list[str],
                       minimum: int) -> list[str]:
    """References a sermon keeps coming back to, apart from its main text.

    Overlapping forms (Genesis 6 and Genesis 6:4) count as one link. A single
    window is not enough when `minimum` is 2, which drops a one-line aside.
    """
    clusters: list[dict[str, Any]] = []
    for refs in per_chunk:
        touched: set[int] = set()
        for ref in refs:
            if primary and bookmap.any_overlap([ref], primary):
                continue
            index = next(
                (i for i, cluster in enumerate(clusters)
                 if bookmap.refs_overlap(ref, str(cluster["ref"]))),
                None,
            )
            if index is None:
                clusters.append({"ref": ref, "count": 0, "span": _ref_span(ref)})
                index = len(clusters) - 1
            if index in touched:
                continue
            touched.add(index)
            clusters[index]["count"] = int(clusters[index]["count"]) + 1
            span = _ref_span(ref)
            if span < int(clusters[index]["span"]):
                clusters[index]["ref"] = ref
                clusters[index]["span"] = span
    return [str(cluster["ref"]) for cluster in clusters if int(cluster["count"]) >= minimum]


def select_scripture_window(rows: list[dict[str, Any]],
                            verse_start: int | None,
                            verse_end: int | None) -> dict[str, Any] | None:
    """Prefer the tightest indexed window that covers the reference."""
    if not rows:
        return None

    def span(row: dict[str, Any]) -> tuple[int, int]:
        start = int(row.get("verse_start") or 0)
        end = int(row.get("verse_end") or 0)
        return end - start, start

    if verse_start is None:
        return max(rows, key=span)

    target_end = verse_end if verse_end is not None else verse_start
    covering = [
        row for row in rows
        if int(row.get("verse_start") or 0) <= verse_start
        and int(row.get("verse_end") or 0) >= target_end
    ]
    if covering:
        return min(covering, key=span)

    overlapping = []
    for row in rows:
        start = int(row.get("verse_start") or 0)
        end = int(row.get("verse_end") or 0)
        if start <= target_end and verse_start <= end:
            overlapping.append(row)
    if overlapping:
        return min(overlapping, key=span)
    return None


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _scripture_for_ref(store, ref: str) -> Passage | None:
    parsed = bookmap.parse_ref(ref)
    if not parsed:
        return None
    osis, chapter, verse_start, verse_end = parsed
    where = f"book_id = {_sql_quote(osis)} AND chapter = {int(chapter)}"
    rows = store.filter_rows(
        "scripture", where,
        columns=["text", "citation", "title", "url", "book_id", "chapter",
                 "verse_start", "verse_end", "source_id"],
    )
    chosen = select_scripture_window(rows, verse_start, verse_end)
    if not chosen or not (chosen.get("text") or "").strip():
        return None
    return Passage(
        collection="scripture",
        text=chosen.get("text", ""),
        citation=chosen.get("citation", "") or chosen.get("title", ""),
        url=chosen.get("url", ""),
        title=chosen.get("title", ""),
        source_id=chosen.get("source_id", ""),
    )


def text_fingerprint(text: str) -> str:
    """Prefix used to drop overlapping windows of the same passage."""
    return (text or "")[:160].strip().lower()


def _grouped_sermons(mosaic_hits: list[Passage]) -> list[list[Passage]]:
    grouped: dict[str, list[Passage]] = {}
    for hit in mosaic_hits:
        grouped.setdefault(hit.source_id, []).append(hit)
    return sorted(
        grouped.values(),
        key=lambda hits: max(hit.score for hit in hits),
        reverse=True,
    )


def _chunk_ref_lists(hits: list[Passage], store) -> list[list[str]]:
    where = f"source_id = {_sql_quote(hits[0].source_id)}"
    rows = store.filter_rows("mosaic", where, columns=["scripture_refs"])
    per_chunk = [bookmap.unpack_refs(row.get("scripture_refs") or "") for row in rows]
    if not per_chunk:
        per_chunk = [hit.scripture_refs for hit in hits]
    return per_chunk


def aside_refs(hits: list[Passage], store, asked_refs: set[str],
               min_chunks: int) -> list[str]:
    """Sustained mentions this sermon should connect for the question.

    A question about the sermon's own passage does not expand its illustrations.
    A question that names an aside, or that matched the window where the aside
    is taught, does.
    """
    primary = hits[0].primary_refs
    sustained = sustained_mentions(_chunk_ref_lists(hits, store), primary, min_chunks)
    question_hits_primary = bool(asked_refs) and bookmap.any_overlap(asked_refs, primary)
    if asked_refs:
        selected = [ref for ref in sustained if bookmap.any_overlap(asked_refs, [ref])]
        if question_hits_primary:
            selected = [ref for ref in selected if not bookmap.any_overlap([ref], primary)]
        return selected
    hit_refs = [ref for hit in hits for ref in hit.scripture_refs]
    return [ref for ref in sustained if bookmap.any_overlap(hit_refs, [ref])]


def bridge_refs(mosaic_hits: list[Passage], store, asked_refs: set[str],
                min_chunks: int) -> list[str]:
    """Sustained passages that can lead to another sermon, highest score first."""
    chosen: list[str] = []
    for hits in _grouped_sermons(mosaic_hits):
        for ref in aside_refs(hits, store, asked_refs, min_chunks):
            if any(bookmap.refs_overlap(ref, existing) for existing in chosen):
                continue
            chosen.append(ref)
    return chosen


def cross_link_passages(mosaic_hits: list[Passage], store, asked_refs: set[str],
                        min_chunks: int, max_passages: int) -> list[Passage]:
    """Scripture windows a sermon connected, capped and ordered by sermon score."""
    if max_passages <= 0:
        return []

    linked: list[Passage] = []
    seen: set[str] = set()

    for hits in _grouped_sermons(mosaic_hits):
        if len(linked) >= max_passages:
            break
        selected = aside_refs(hits, store, asked_refs, min_chunks)
        if not selected:
            continue

        targets = list(selected)
        for ref in hits[0].primary_refs:
            if not bookmap.any_overlap([ref], targets):
                targets.append(ref)

        sermon = hits[0].citation or hits[0].label
        connection = f"Connected in {sermon}"
        for ref in targets:
            if len(linked) >= max_passages:
                break
            passage = _scripture_for_ref(store, ref)
            if passage is None:
                continue
            key = passage.citation or passage.text[:80]
            if key in seen:
                continue
            seen.add(key)
            passage.connection = connection
            passage.score = hits[0].score
            linked.append(passage)

    return linked


def _chunk_for_bridge(chunks: list[dict[str, Any]], ref: str) -> dict[str, Any] | None:
    """The window that quotes the bridge, or the sermon's first window."""
    if not chunks:
        return None
    quoting = [
        chunk for chunk in chunks
        if bookmap.any_overlap(
            [ref], bookmap.unpack_refs(chunk.get("scripture_refs") or ""),
        )
    ]
    return quoting[0] if quoting else chunks[0]


def related_sermon_passages(mosaic_hits: list[Passage], store, bridges: list[str],
                            max_sermons: int) -> list[Passage]:
    """Other sermons whose main text is a bridge passage.

    One hop from the sermons already retrieved. Their own asides are not followed.
    A sermon that only mentions the passage does not qualify.
    """
    if max_sermons <= 0 or not bridges:
        return []

    known = {hit.source_id for hit in mosaic_hits if hit.source_id}
    rows = store.filter_rows(
        "mosaic",
        "source_id != ''",
        columns=["source_id", "primary_refs", "citation", "title", "date",
                 "speaker", "url"],
        limit=None,
    )

    chosen: list[tuple[str, str]] = []
    seen = set(known)
    for ref in bridges:
        matches: list[dict[str, Any]] = []
        matched_ids: set[str] = set()
        for row in rows:
            source_id = row.get("source_id") or ""
            if not source_id or source_id in seen or source_id in matched_ids:
                continue
            primary = bookmap.unpack_refs(row.get("primary_refs") or "")
            if not bookmap.any_overlap([ref], primary):
                continue
            matched_ids.add(source_id)
            matches.append(row)
        matches.sort(key=lambda row: (
            (row.get("citation") or row.get("title") or ""),
            row.get("source_id") or "",
        ))
        for row in matches:
            source_id = row.get("source_id") or ""
            seen.add(source_id)
            chosen.append((ref, source_id))
            if len(chosen) >= max_sermons:
                break
        if len(chosen) >= max_sermons:
            break

    linked: list[Passage] = []
    for ref, source_id in chosen:
        where = f"source_id = {_sql_quote(source_id)}"
        chunks = store.filter_rows(
            "mosaic", where,
            columns=["text", "citation", "title", "url", "date", "speaker",
                     "source_id", "primary_refs", "scripture_refs"],
            limit=None,
        )
        chunk = _chunk_for_bridge(chunks, ref)
        if chunk is None or not (chunk.get("text") or "").strip():
            continue
        linked.append(Passage(
            collection="mosaic",
            text=chunk.get("text", ""),
            citation=chunk.get("citation", "") or chunk.get("title", ""),
            url=chunk.get("url", ""),
            title=chunk.get("title", ""),
            date=chunk.get("date", ""),
            speaker=chunk.get("speaker", ""),
            source_id=source_id,
            primary_refs=bookmap.unpack_refs(chunk.get("primary_refs", "")),
            scripture_refs=bookmap.unpack_refs(chunk.get("scripture_refs", "")),
            connection=f"Also teaches {ref}",
        ))
    return linked


EXPANSION_SYSTEM = """\
You prepare searches over sermon transcripts. The question may use words the \
sermons never use. Reply with up to {n} short search phrases for the same \
subject, in the biblical and pastoral wording a sermon would actually use. \
One phrase per line. No numbering and no explanation.\
"""


def _strip_list_marker(line: str) -> str:
    stripped = line.strip().lstrip("-*•").strip()
    index = 0
    while index < len(stripped) and stripped[index].isdigit():
        index += 1
    if index and index < len(stripped) and stripped[index] in ".)":
        return stripped[index + 1:].strip()
    return stripped


def expansion_request(question: str, max_queries: int) -> list[dict[str, str]]:
    """The messages that ask for sermon-worded searches. No topic list."""
    return [
        {"role": "system", "content": EXPANSION_SYSTEM.format(n=max_queries)},
        {"role": "user", "content": question.strip()},
    ]


def parse_expansion_lines(text: str, question: str, max_queries: int) -> list[str]:
    """Turn the model's lines into search phrases.

    The question itself is dropped. Further lines past the cap are ignored.
    """
    if max_queries <= 0:
        return []
    folded_question = " ".join((question or "").casefold().split())
    phrases: list[str] = []
    seen = {folded_question} if folded_question else set()
    for raw in (text or "").splitlines():
        line = _strip_list_marker(raw)
        if not line or len(line) > 200:
            continue
        key = " ".join(line.casefold().split())
        if not key or key in seen:
            continue
        seen.add(key)
        phrases.append(line)
        if len(phrases) >= max_queries:
            break
    return phrases


def expansion_phrases(chat, question: str, max_queries: int) -> list[str]:
    """Ask the local model how a sermon would word this subject.

    Same step for every question. Nothing is special-cased by topic.
    """
    if max_queries <= 0 or not (question or "").strip():
        return []
    text = chat.complete(
        expansion_request(question, max_queries),
        max_tokens=96,
        temperature=0,
    )
    return parse_expansion_lines(text, question, max_queries)


def merge_expansion(existing: list[Passage], extra: list[Passage],
                    max_new: int) -> list[Passage]:
    """Append up to `max_new` passages whose text is not already present.

    Direct hits stay in place. A duplicate keeps the earlier copy.
    """
    if max_new <= 0:
        return list(existing)
    seen = {text_fingerprint(passage.text) for passage in existing}
    merged = list(existing)
    added = 0
    for passage in extra:
        fingerprint = text_fingerprint(passage.text)
        if not fingerprint or fingerprint in seen:
            continue
        seen.add(fingerprint)
        merged.append(passage)
        added += 1
        if added >= max_new:
            break
    return merged


class Retriever:
    def __init__(self, settings: dict[str, Any], index_dir, *,
                 device: str | None = None, verify: bool = True) -> None:
        self.settings = settings
        self.model = EmbeddingModel(settings, device=device)
        self.identity = self.model.identity()
        self.store = VectorStore(index_dir, self.identity.dimension)

        if verify:
            # A mismatch here produces no error and no obvious symptom, only
            # steadily irrelevant answers, so it is checked up front.
            self.store.assert_embedding_matches(self.identity)

        cfg = settings["Retrieval"]
        self.top_k: dict[str, int] = cfg["TopK"]
        self.weights: dict[str, float] = cfg["Weights"]
        self.multiplier = int(cfg.get("CandidateMultiplier", 4))
        self.max_context = int(cfg.get("MaxContextChars", 14000))
        self.primary_boost = float(cfg.get("PrimaryRefBoost", cfg.get("ScriptureRefBoost", 0.25)))
        self.mentioned_boost = float(cfg.get("MentionedRefBoost", 0.10))
        self.cross_link_min_chunks = int(cfg.get("CrossLinkMinChunks", 2))
        self.cross_link_max = int(cfg.get("CrossLinkMaxPassages", 2))
        self.cross_link_max_sermons = int(cfg.get("CrossLinkMaxRelatedSermons", 3))
        self.expansion_max_queries = int(cfg.get("ExpansionMaxQueries", 3))
        self.expansion_max_passages = int(cfg.get("ExpansionMaxPassages", 4))

    def search(self, question: str, expansions: list[str] | None = None) -> RetrievalResult:
        vector = self.model.encode_query(question)
        asked_refs = set(bookmap.extract_refs(question))

        result = RetrievalResult(question=question)
        result.counts = self.store.counts()

        for collection in COLLECTIONS:
            want = int(self.top_k.get(collection, 4))
            if want <= 0:
                continue

            rows = self.store.search(collection, vector, want * self.multiplier)
            scored = [self._passage_from_row(collection, row, asked_refs) for row in rows]
            scored.sort(key=lambda p: p.score, reverse=True)
            result.passages.extend(self._dedupe(scored)[:want])

        result.passages.sort(key=lambda p: p.score, reverse=True)
        extra = self._expansion_hits(expansions or [], asked_refs)
        result.passages = merge_expansion(
            result.passages, extra, self.expansion_max_passages,
        )
        result.passages.sort(key=lambda p: p.score, reverse=True)
        result.passages = self._fit_budget(result.passages)
        self._attach_cross_links(result, asked_refs)
        return result

    def _passage_from_row(self, collection: str, row: dict[str, Any],
                          asked_refs: set[str]) -> Passage:
        raw = float(row.get("score", 0.0))
        score = raw * float(self.weights.get(collection, 1.0))
        # Embeddings miss exact verse references. A sermon about the
        # asked passage outranks one that only mentions it.
        score += ref_boost_amount(
            collection,
            row.get("citation", ""),
            row.get("primary_refs", ""),
            row.get("scripture_refs", ""),
            asked_refs,
            self.primary_boost,
            self.mentioned_boost,
        )
        return Passage(
            collection=collection,
            text=row.get("text", ""),
            citation=row.get("citation", "") or row.get("title", ""),
            url=row.get("url", ""),
            title=row.get("title", ""),
            date=row.get("date", ""),
            speaker=row.get("speaker", ""),
            score=score,
            raw_score=raw,
            source_id=row.get("source_id", ""),
            primary_refs=bookmap.unpack_refs(row.get("primary_refs", "")),
            scripture_refs=bookmap.unpack_refs(row.get("scripture_refs", "")),
        )

    def _expansion_hits(self, expansions: list[str], asked_refs: set[str]) -> list[Passage]:
        """Second mosaic search using phrases the model proposed for this question."""
        phrases: list[str] = []
        seen: set[str] = set()
        for phrase in expansions:
            text = str(phrase).strip()
            key = " ".join(text.casefold().split())
            if not key or key in seen:
                continue
            seen.add(key)
            phrases.append(text)
            if len(phrases) >= self.expansion_max_queries:
                break
        if not phrases:
            return []
        want = int(self.top_k.get("mosaic", 4))
        found: list[Passage] = []
        for phrase in phrases:
            vector = self.model.encode_query(phrase)
            rows = self.store.search("mosaic", vector, max(want, 1) * self.multiplier)
            found.extend(
                self._passage_from_row("mosaic", row, asked_refs) for row in rows
            )
        found.sort(key=lambda p: p.score, reverse=True)
        return self._dedupe(found)

    def _attach_cross_links(self, result: RetrievalResult, asked_refs: set[str]) -> None:
        """Pull connected Scripture and other sermons on that same passage.

        A question about the sermon's own passage does not expand that sermon's
        illustrations. Related sermons are one hop: their own asides are not followed.
        """
        if self.cross_link_max <= 0 and self.cross_link_max_sermons <= 0:
            return
        mosaic = [
            p for p in result.passages
            if p.collection == "mosaic" and p.source_id and not p.connection
        ]
        if not mosaic:
            return
        bridges = bridge_refs(
            mosaic, self.store, asked_refs, self.cross_link_min_chunks,
        )
        linked: list[Passage] = []
        if self.cross_link_max > 0:
            linked = cross_link_passages(
                mosaic, self.store, asked_refs,
                self.cross_link_min_chunks, self.cross_link_max,
            )
        related: list[Passage] = []
        if self.cross_link_max_sermons > 0:
            related = related_sermon_passages(
                mosaic, self.store, bridges, self.cross_link_max_sermons,
            )
        used = sum(len(p.text) + len(p.label) + len(p.connection) + 32
                   for p in result.passages)
        scripture_kept = 0
        for passage in linked + related:
            if passage.collection == "scripture" and scripture_kept >= self.cross_link_max:
                continue
            cost = len(passage.text) + len(passage.label) + len(passage.connection) + 32
            if used + cost > self.max_context:
                continue
            result.passages.append(passage)
            used += cost
            if passage.collection == "scripture":
                scripture_kept += 1

    @staticmethod
    def _dedupe(passages: list[Passage]) -> list[Passage]:
        """Drop near-identical text, which overlapping windows produce."""
        seen: set[str] = set()
        out: list[Passage] = []
        for passage in passages:
            fingerprint = text_fingerprint(passage.text)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            out.append(passage)
        return out

    def _fit_budget(self, passages: list[Passage]) -> list[Passage]:
        """Trim to the context budget, keeping at least one of each source."""
        kept: list[Passage] = []
        used = 0
        represented: set[str] = set()

        for passage in passages:
            cost = len(passage.text) + len(passage.label) + 32
            first_of_kind = passage.collection not in represented

            if used + cost > self.max_context and not first_of_kind:
                continue
            if used + cost > self.max_context and first_of_kind and used > self.max_context:
                continue

            kept.append(passage)
            represented.add(passage.collection)
            used += cost

        return kept
