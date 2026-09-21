"""Normalise the downloaded Bible JSON into verse rows.

Three source shapes are accepted, because public-domain fallbacks rarely use
the same layout as the configured source:

  A. nested object   {"Genesis": {"1": {"1": "...", "2": "..."}}}
  B. JSONL           one {"book","chapter","verse","text"} object per line
  C. array of books  [{"name"|"abbrev": ..., "chapters": [["v1","v2"], ...]}]

Whatever the shape, the output is the same: verse rows keyed by OSIS id, so
citations survive a change of source.

Validation is by book count. A "does Genesis exist" check would accept a file
that stops partway through and leave an index that looks healthy and answers
nothing for everything after the cut.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from lib import bookmap
from lib.bookmap import UnknownBookError
from lib.state import atomic_write_json
from workers.common import WorkerContext, base_parser, log

VerseRow = dict[str, Any]


def _rows_from_nested(payload: dict) -> Iterator[VerseRow]:
    for book_name, chapters in payload.items():
        if not isinstance(chapters, dict):
            continue
        osis = bookmap.to_osis(book_name)
        for chapter_key, verses in chapters.items():
            if not isinstance(verses, dict):
                continue
            try:
                chapter = int(str(chapter_key).strip())
            except ValueError:
                continue
            for verse_key, text in verses.items():
                try:
                    verse = int(str(verse_key).strip())
                except ValueError:
                    continue
                if isinstance(text, str) and text.strip():
                    yield {"book_id": osis, "chapter": chapter,
                           "verse": verse, "text": text.strip()}


def _rows_from_array(payload: list) -> Iterator[VerseRow]:
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("book") or entry.get("abbrev") or ""
        osis = bookmap.to_osis(str(name))
        for chapter_index, verses in enumerate(entry.get("chapters") or [], start=1):
            for verse_index, text in enumerate(verses or [], start=1):
                if isinstance(text, str) and text.strip():
                    yield {"book_id": osis, "chapter": chapter_index,
                           "verse": verse_index, "text": text.strip()}


def _rows_from_jsonl(path: Path) -> Iterator[VerseRow]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            osis = row.get("book_id") or bookmap.to_osis(str(row.get("book", "")))
            text = (row.get("text") or "").strip()
            if not text:
                continue
            yield {
                "book_id": osis,
                "chapter": int(row["chapter"]),
                "verse": int(row["verse"]),
                "verse_end": int(row["verse_end"]) if row.get("verse_end") else None,
                "text": text,
            }


def load_rows(path: Path) -> list[VerseRow]:
    if path.suffix.lower() == ".jsonl":
        return list(_rows_from_jsonl(path))

    # Explicit UTF-8. The source is full of curly quotes and dashes that a
    # locale-default read would corrupt.
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)

    if isinstance(payload, dict):
        return list(_rows_from_nested(payload))
    if isinstance(payload, list):
        return list(_rows_from_array(payload))
    raise ValueError(f"Unrecognised Bible JSON shape in {path}")


def find_source(ctx: WorkerContext) -> Path | None:
    configured = ctx.root / ctx.settings["Bible"]["LocalFile"].replace("/", "\\") \
        if "\\" in str(ctx.root) else ctx.root / ctx.settings["Bible"]["LocalFile"]
    if configured.is_file():
        return configured

    for pattern in ("*.json", "*.jsonl"):
        for candidate in sorted(ctx.paths.bible.glob(pattern)):
            return candidate
    return None


def main() -> int:
    parser = base_parser("Normalise the Bible JSON into verse rows")
    args = parser.parse_args()

    with WorkerContext(args) as ctx:
        cfg = ctx.settings["Bible"]
        expected = int(cfg.get("ExpectedBookCount", 66))
        translation = cfg.get("Translation", "ESV")
        target = ctx.paths.bible / "verses.jsonl"

        if target.exists() and not ctx.force:
            log(f"Verse rows already prepared: {ctx.relative(target)}")
            return 0

        source = find_source(ctx)
        if source is None:
            log("No Bible source found in data/bible/.")
            log("Run: .\\Mosaic-NightJob.ps1 -Action EnsureDeps")
            log("The rest of the pipeline still works; scripture will be empty.")
            return 0

        log(f"Reading {ctx.relative(source)}")

        try:
            rows = load_rows(source)
        except UnknownBookError as exc:
            # Loud by design. Dropping a book silently would leave an index
            # that looks complete and answers nothing for that book.
            log(f"ERROR: {exc}")
            return 2
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log(f"ERROR: {source.name} is not readable JSON: {exc}")
            log("A truncated download is the usual cause. Delete the file and "
                "re-run EnsureDeps.")
            return 2

        books = {row["book_id"] for row in rows}
        log(f"Parsed {len(rows):,} verses across {len(books)} books.")

        if len(books) < expected:
            missing = [
                bookmap.display_name(osis)
                for osis, _ in bookmap.CANON if osis not in books
            ]
            log(f"ERROR: expected {expected} books, found {len(books)}.")
            log(f"Missing: {', '.join(missing[:12])}"
                f"{' ...' if len(missing) > 12 else ''}")
            log("This is a partial source. Delete data/bible/ and re-run "
                "EnsureDeps rather than indexing an incomplete Bible.")
            return 2

        rows.sort(key=lambda r: (bookmap.sort_key(r["book_id"]),
                                 r["chapter"], r["verse"]))

        tmp = target.with_suffix(".jsonl.part")
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                row["translation"] = translation
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(target)

        atomic_write_json(ctx.paths.bible / "bible-meta.json", {
            "source": source.name,
            "translation": translation,
            "books": len(books),
            "verses": len(rows),
        })

        log(f"Wrote {ctx.relative(target)} ({len(rows):,} verses, {translation}).")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
