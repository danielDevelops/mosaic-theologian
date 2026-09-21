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
        self.ref_boost = float(cfg.get("ScriptureRefBoost", 0.15))

    def search(self, question: str) -> RetrievalResult:
        vector = self.model.encode_query(question)
        asked_refs = set(bookmap.extract_refs(question))

        result = RetrievalResult(question=question)
        result.counts = self.store.counts()

        for collection in COLLECTIONS:
            want = int(self.top_k.get(collection, 4))
            if want <= 0:
                continue

            rows = self.store.search(collection, vector, want * self.multiplier)
            weight = float(self.weights.get(collection, 1.0))
            scored: list[Passage] = []

            for row in rows:
                raw = float(row.get("score", 0.0))
                score = raw * weight

                # Embeddings are weak on exact tokens like verse references,
                # so an explicit reference in the question gets a direct boost.
                if asked_refs:
                    row_refs = set((row.get("scripture_refs") or "").split())
                    citation = row.get("citation", "")
                    if any(ref.replace(" ", "") in citation.replace(" ", "")
                           for ref in asked_refs) or (asked_refs & row_refs):
                        score += self.ref_boost

                scored.append(Passage(
                    collection=collection,
                    text=row.get("text", ""),
                    citation=row.get("citation", "") or row.get("title", ""),
                    url=row.get("url", ""),
                    title=row.get("title", ""),
                    date=row.get("date", ""),
                    speaker=row.get("speaker", ""),
                    score=score,
                    raw_score=raw,
                ))

            scored.sort(key=lambda p: p.score, reverse=True)
            result.passages.extend(self._dedupe(scored)[:want])

        result.passages.sort(key=lambda p: p.score, reverse=True)
        result.passages = self._fit_budget(result.passages)
        return result

    @staticmethod
    def _dedupe(passages: list[Passage]) -> list[Passage]:
        """Drop near-identical text, which overlapping windows produce."""
        seen: set[str] = set()
        out: list[Passage] = []
        for passage in passages:
            fingerprint = passage.text[:160].strip().lower()
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
