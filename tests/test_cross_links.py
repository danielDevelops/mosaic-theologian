"""Sermon cross-links: primary passage, sustained asides, and labeled hops.

These tests stay off the embedding model and off LanceDB. They exercise the
decisions the retriever makes once a sermon has been chunked.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import bookmap, chunking
from lib.prompt import SYSTEM_PROMPT, build_user_message
from lib.retrieval import (
    Passage,
    RetrievalResult,
    bridge_refs,
    cross_link_passages,
    EXPANSION_SYSTEM,
    expansion_request,
    fit_context,
    parse_expansion_lines,
    merge_expansion,
    ref_boost_amount,
    related_sermon_passages,
    select_scripture_window,
    sustained_mentions,
)
from workers.index_build import replace_index


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(1)


def test_refs_round_trip() -> None:
    packed = bookmap.pack_refs(["1 John 3:1-3", "Genesis 6:4"])
    check("pack keeps book names", packed == "1 John 3:1-3\nGenesis 6:4", packed)
    check("unpack restores both",
          bookmap.unpack_refs(packed) == ["1 John 3:1-3", "Genesis 6:4"])
    check("chapter overlaps its verses",
          bookmap.refs_overlap("1 John 3", "1 John 3:1-3"))
    check("different books do not overlap",
          not bookmap.refs_overlap("1 John 3:1", "Genesis 6:4"))
    check("ranges in one chapter overlap",
          bookmap.refs_overlap("1 John 3:1-3", "1 John 3:2"))


def test_primary_stamp() -> None:
    filler = "The Father has loved us with a lasting love. "
    text = (
        "Open your Bible to 1 John 3:1. "
        + filler * 12
        + "Someone will ask about the Nephilim in Genesis 6:4. "
        + "Genesis 6:4 is the line they quote back."
    )
    chunks = chunking.chunk_sermon(
        source_id="sermon-1",
        text=text,
        title="Children of God",
        url="https://example.test/sermon",
        date="2020-05-01",
        speaker="Ada",
        target_tokens=30,
        overlap_tokens=0,
    )
    check("sermon split into windows", len(chunks) >= 3, f"{len(chunks)} chunks")
    check("opening passage stamped on every window",
          all(chunk.primary_refs == ["1 John 3:1"] for chunk in chunks),
          str([chunk.primary_refs for chunk in chunks]))
    check("later window keeps its own mention",
          "Genesis 6:4" in chunks[-1].scripture_refs,
          str(chunks[-1].scripture_refs))
    check("mention is not rewritten as the primary",
          chunks[-1].primary_refs == ["1 John 3:1"])

    forced = chunking.chunk_sermon(
        source_id="sermon-2",
        text="The Nephilim of Genesis 6:4 come up first. Then we read on.",
        title="Children of God",
        url="https://example.test/sermon",
        primary_refs=["1 John 3:1-3"],
    )
    check("page passage wins over the transcript opening",
          all(chunk.primary_refs == ["1 John 3:1-3"] for chunk in forced))

    from_page = chunking.sermon_primary_refs(
        title="Children of God",
        page_refs=["1 John 3:1-3"],
        page_title="",
        job_refs=["Genesis 6:4"],
    )
    check("page refs are kept with the job refs",
          from_page == ["1 John 3:1-3", "Genesis 6:4"], str(from_page))


def test_sustained_filter() -> None:
    primary = ["1 John 3:1-3"]
    once = [["1 John 3:1-3"], ["Genesis 6:4"], ["1 John 3:2"]]
    check("one aside window is not a cross-link",
          sustained_mentions(once, primary, 2) == [])
    twice = [["1 John 3:1"], ["Genesis 6"], ["Genesis 6:4"], ["1 John 3:2"]]
    check("an aside that returns is a cross-link",
          sustained_mentions(twice, primary, 2) == ["Genesis 6:4"],
          str(sustained_mentions(twice, primary, 2)))
    check("the sermon's own passage is not an aside",
          "1 John" not in " ".join(sustained_mentions(twice, primary, 2)))


def test_boosts() -> None:
    asked = {"1 John 3:1-3"}
    primary = ref_boost_amount(
        "mosaic", "Children of God | Ada | 2020-05-01",
        bookmap.pack_refs(["1 John 3:1-3"]),
        bookmap.pack_refs(["Genesis 6:4"]),
        asked, 0.25, 0.1,
    )
    mentioned = ref_boost_amount(
        "mosaic", "Another sermon",
        "",
        bookmap.pack_refs(["Genesis 6:4", "1 John 3:16"]),
        {"1 John 3:16"}, 0.25, 0.1,
    )
    check("primary boost is the larger one", primary == 0.25 and mentioned == 0.1,
          f"primary={primary} mentioned={mentioned}")


class MemoryIndex:
    def __init__(self, mosaic: list[dict], scripture: list[dict]) -> None:
        self.mosaic = mosaic
        self.scripture = scripture

    def filter_rows(self, collection: str, where: str, columns=None, limit: int | None = 500):
        if collection == "mosaic":
            if where.startswith("source_id = "):
                source = where.split("'")[1]
                rows = [row for row in self.mosaic if row["source_id"] == source]
            else:
                rows = [row for row in self.mosaic if row.get("source_id")]
        else:
            book = where.split("'")[1]
            chapter = int(where.rsplit("=", 1)[-1].strip())
            rows = [
                row for row in self.scripture
                if row["book_id"] == book and int(row["chapter"]) == chapter
            ]
        if limit is not None:
            rows = rows[:limit]
        return rows


def _sermon(source: str, primary: list[str], mentioned: list[str], score: float = 0.8) -> Passage:
    return Passage(
        collection="mosaic",
        text="sermon window",
        citation="Children of God | Ada | 2020-05-01",
        title="Children of God",
        date="2020-05-01",
        speaker="Ada",
        score=score,
        source_id=source,
        primary_refs=list(primary),
        scripture_refs=list(mentioned),
    )


def _scripture_rows() -> list[dict]:
    return [
        {
            "book_id": "1JN", "chapter": 3, "verse_start": 1, "verse_end": 24,
            "text": "chapter of 1 John 3",
            "citation": "1 John 3 (ESV)", "title": "1 John 3", "url": "", "source_id": "bible:1JN",
        },
        {
            "book_id": "1JN", "chapter": 3, "verse_start": 1, "verse_end": 6,
            "text": "1 John 3:1-6 text",
            "citation": "1 John 3:1-6 (ESV)", "title": "1 John 3:1-6", "url": "", "source_id": "bible:1JN",
        },
        {
            "book_id": "GEN", "chapter": 6, "verse_start": 1, "verse_end": 8,
            "text": "Genesis 6:1-8 text",
            "citation": "Genesis 6:1-8 (ESV)", "title": "Genesis 6:1-8", "url": "", "source_id": "bible:GEN",
        },
        {
            "book_id": "GEN", "chapter": 6, "verse_start": 1, "verse_end": 22,
            "text": "chapter of Genesis 6",
            "citation": "Genesis 6 (ESV)", "title": "Genesis 6", "url": "", "source_id": "bible:GEN",
        },
    ]


def test_hops() -> None:
    mosaic_rows = [
        {"source_id": "sermon-1", "scripture_refs": bookmap.pack_refs(["1 John 3:1-3"])},
        {"source_id": "sermon-1", "scripture_refs": bookmap.pack_refs(["Genesis 6:4"])},
        {"source_id": "sermon-1", "scripture_refs": bookmap.pack_refs(["Genesis 6:4"])},
    ]
    store = MemoryIndex(mosaic_rows, _scripture_rows())
    primary_hit = [_sermon("sermon-1", ["1 John 3:1-3"], ["1 John 3:1-3"])]
    about_the_text = cross_link_passages(primary_hit, store, {"1 John 3:1-3"}, 2, 2)
    check("asking the main text does not pull the aside", about_the_text == [])

    aside_hit = [_sermon("sermon-1", ["1 John 3:1-3"], ["Genesis 6:4"])]
    hopped = cross_link_passages(aside_hit, store, {"Genesis 6:4"}, 2, 2)
    labels = [p.citation for p in hopped]
    check("an aside pulls its own text and the sermon's text",
          labels == ["Genesis 6:1-8 (ESV)", "1 John 3:1-6 (ESV)"], str(labels))
    check("both hops name the sermon",
          all(p.connection == "Connected in Children of God | Ada | 2020-05-01" for p in hopped),
          str([p.connection for p in hopped]))

    topical = cross_link_passages(aside_hit, store, set(), 2, 2)
    check("a topical question uses the matched window's aside",
          [p.citation for p in topical] == labels, str([p.citation for p in topical]))

    exposition = [_sermon("sermon-1", ["1 John 3:1-3"], ["1 John 3:1-3"])]
    check("a matched window without the aside does not hop",
          cross_link_passages(exposition, store, set(), 2, 2) == [])

    tight = select_scripture_window(_scripture_rows()[:2], 1, 3)
    check("verse range prefers the tight window",
          tight is not None and tight["citation"] == "1 John 3:1-6 (ESV)")


def _related_store() -> MemoryIndex:
    mosaic_rows = [
        {"source_id": "sermon-1", "scripture_refs": bookmap.pack_refs(["1 John 3:1-3"])},
        {"source_id": "sermon-1", "scripture_refs": bookmap.pack_refs(["Genesis 6:4"])},
        {"source_id": "sermon-1", "scripture_refs": bookmap.pack_refs(["Genesis 6:4"])},
        {
            "source_id": "sermon-2",
            "primary_refs": bookmap.pack_refs(["Genesis 6:1-8"]),
            "scripture_refs": bookmap.pack_refs(["Romans 8:28"]),
            "text": "opening of the flood sermon",
            "citation": "Flood | Bo | 2021-01-01",
            "title": "Flood",
            "date": "2021-01-01",
            "speaker": "Bo",
            "url": "https://example.test/flood",
        },
        {
            "source_id": "sermon-2",
            "primary_refs": bookmap.pack_refs(["Genesis 6:1-8"]),
            "scripture_refs": bookmap.pack_refs(["Genesis 6:4"]),
            "text": "the nephilim window",
            "citation": "Flood | Bo | 2021-01-01",
            "title": "Flood",
            "date": "2021-01-01",
            "speaker": "Bo",
            "url": "https://example.test/flood",
        },
        {
            "source_id": "sermon-mention",
            "primary_refs": bookmap.pack_refs(["Romans 1:16"]),
            "scripture_refs": bookmap.pack_refs(["Genesis 6:4"]),
            "text": "a passing mention of the nephilim",
            "citation": "Gospel | Cy | 2022-01-01",
            "title": "Gospel",
        },
        {
            "source_id": "sermon-3",
            "primary_refs": bookmap.pack_refs(["Romans 8:28"]),
            "scripture_refs": bookmap.pack_refs(["Romans 8:28"]),
            "text": "all things work together",
            "citation": "Called | Dee | 2023-01-01",
            "title": "Called",
        },
        {
            "source_id": "sermon-beta",
            "primary_refs": bookmap.pack_refs(["Genesis 6:4"]),
            "scripture_refs": bookmap.pack_refs(["Genesis 6:4"]),
            "text": "beta window",
            "citation": "Beta | Ed | 2019-01-01",
            "title": "Beta",
        },
    ]
    return MemoryIndex(mosaic_rows, _scripture_rows())


def test_related_sermons() -> None:
    store = _related_store()
    aside_hit = [_sermon("sermon-1", ["1 John 3:1-3"], ["Genesis 6:4"])]
    bridges = bridge_refs(aside_hit, store, {"Genesis 6:4"}, 2)
    check("the aside is the bridge", bridges == ["Genesis 6:4"], str(bridges))

    related = related_sermon_passages(aside_hit, store, bridges, 3)
    sources = [p.source_id for p in related]
    check("the originating sermon is not repeated", "sermon-1" not in sources, str(sources))
    check("a mention-only sermon is skipped", "sermon-mention" not in sources, str(sources))
    check("a sermon reached only through the related sermon is skipped",
          "sermon-3" not in sources, str(sources))
    check("sermons on the bridge passage are included",
          sources == ["sermon-beta", "sermon-2"], str(sources))
    check("the join is the bridge passage",
          all(p.connection == "Also teaches Genesis 6:4" for p in related),
          str([p.connection for p in related]))
    flood = next(p for p in related if p.source_id == "sermon-2")
    check("the quoting window is the one attached",
          flood.text == "the nephilim window", flood.text)

    capped = related_sermon_passages(aside_hit, store, bridges, 1)
    check("the related-sermon cap holds",
          [p.source_id for p in capped] == ["sermon-beta"],
          str([p.source_id for p in capped]))

    about_the_text = [_sermon("sermon-1", ["1 John 3:1-3"], ["1 John 3:1-3"])]
    check("the main text does not open a sermon hop",
          bridge_refs(about_the_text, store, {"1 John 3:1-3"}, 2) == [])


def test_expansion() -> None:
    settings = json.loads(
        (Path(__file__).resolve().parent.parent / "config" / "settings.json").read_text(
            encoding="utf-8",
        )
    )
    instruction = EXPANSION_SYSTEM.casefold()
    check("settings carry no topic list",
          "Expansions" not in settings["Retrieval"]
          and "arminian" not in instruction
          and "abortion" not in instruction
          and "exposure" not in instruction)
    check("the expansion restates the act with no verdict",
          "concrete act" in instruction and "no verdict" in instruction)
    question = "How should we talk about this?"
    request = expansion_request(question, 3)
    check("the expansion request is the question plus a general instruction",
          request[-1]["content"] == question
          and "sermon" in request[0]["content"].casefold())

    parsed = parse_expansion_lines(
        "\n".join([
            "1. first wording a sermon might use",
            "- second wording",
            question,
            "third wording",
            "fourth wording",
        ]),
        question,
        3,
    )
    check("lines become phrases, the question is dropped, and the cap holds",
          parsed == [
              "first wording a sermon might use",
              "second wording",
              "third wording",
          ],
          str(parsed))
    check("no lines means no extra search",
          parse_expansion_lines("", question, 3) == [])

    direct = [
        Passage(collection="mosaic", text="already here", citation="Direct", score=0.9),
        Passage(collection="mosaic", text="second direct", citation="Second", score=0.8),
    ]
    extra = [
        Passage(collection="mosaic", text="already here", citation="Duplicate", score=0.95),
        Passage(collection="mosaic", text="election sermon", citation="Election", score=0.7),
        Passage(collection="mosaic", text="grace sermon", citation="Grace", score=0.6),
    ]
    merged = merge_expansion(direct, extra, 1)
    check("expansion keeps direct hits and drops duplicates",
          [p.citation for p in merged] == ["Direct", "Second", "Election"],
          str([p.citation for p in merged]))

    already = Passage(
        collection="scripture", text="a man lies with a male",
        citation="Leviticus 18:22", score=0.4,
    )
    again = Passage(
        collection="scripture", text="a man lies with a male",
        citation="Leviticus 18:22", score=0.8, from_expansion=True,
    )
    reserved = merge_expansion([already], [again], 3)
    check("a scripture hit found again by the act search is reserved",
          reserved[0].from_expansion is True and len(reserved) == 1)

    sermon = Passage(
        collection="mosaic", text="word " * 2000, citation="Sermon", score=0.99,
    )
    nearby = Passage(
        collection="scripture", text="a wedding", citation="John 2:1", score=0.95,
    )
    act = Passage(
        collection="scripture", text="a man lies with a male",
        citation="Leviticus 18:22", score=0.1, from_expansion=True,
    )
    kept = fit_context([sermon, nearby, act], 5000)
    check("act-matched scripture is kept when a sermon fills the budget",
          [p.citation for p in kept] == ["Leviticus 18:22", "Sermon"],
          str([p.citation for p in kept]))


def test_prompt_labels_links_under_teaching() -> None:
    result = RetrievalResult(question="What about the Nephilim?")
    result.passages = [
        Passage(collection="scripture", text="See what kind of love.",
                citation="1 John 3:1-3 (ESV)"),
        Passage(collection="mosaic", text="He mentions the Nephilim here.",
                citation="Children of God | Ada | 2020-05-01",
                title="Children of God", date="2020-05-01"),
        Passage(collection="scripture", text="The Nephilim were on the earth.",
                citation="Genesis 6:1-4 (ESV)",
                connection="Connected in Children of God | Ada | 2020-05-01"),
        Passage(collection="mosaic", text="the nephilim window",
                citation="Flood | Bo | 2021-01-01",
                title="Flood", date="2021-01-01",
                connection="Also teaches Genesis 6:4"),
    ]
    message = build_user_message(result)
    scripture, teaching = message.split("### CHURCH TEACHING", 1)
    check("main scripture stays in the scripture block",
          "See what kind of love." in scripture and "Nephilim were on the earth" not in scripture)
    check("the link is under church teaching",
          "Connected in Children of God | Ada | 2020-05-01" in teaching
          and "The Nephilim were on the earth." in teaching)
    check("a related sermon is listed once",
          teaching.count("the nephilim window") == 1
          and "Also teaches Genesis 6:4" in teaching)
    check("the answer is one piece, not three headings",
          "What Scripture says" not in SYSTEM_PROMPT
          and "Wider Christian thought" not in SYSTEM_PROMPT
          and "Do not write a sources list" in SYSTEM_PROMPT)
    check("a modern word is answered from the act in the passages",
          "modern word" in SYSTEM_PROMPT
          and "Do not fill that gap from general knowledge." in SYSTEM_PROMPT
          and "Do not add an outside position the passages do not contain." in SYSTEM_PROMPT)


def test_replace_index() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        live = root / "index"
        side = root / "index.rebuilding"
        live.mkdir()
        (live / "keep").write_text("old", encoding="utf-8")
        try:
            replace_index(live, side)
        except OSError:
            pass
        check("failed swap restores the live index",
              (live / "keep").read_text(encoding="utf-8") == "old")

        side.mkdir()
        (side / "new").write_text("new", encoding="utf-8")
        replace_index(live, side)
        check("finished rebuild replaces the live index",
              (live / "new").read_text(encoding="utf-8") == "new"
              and not (live / "keep").exists()
              and not (root / "index.prev").exists()
              and not side.exists())


def main() -> None:
    test_refs_round_trip()
    test_primary_stamp()
    test_sustained_filter()
    test_boosts()
    test_hops()
    test_related_sermons()
    test_expansion()
    test_prompt_labels_links_under_teaching()
    test_replace_index()
    print("cross-link tests passed")


if __name__ == "__main__":
    main()
