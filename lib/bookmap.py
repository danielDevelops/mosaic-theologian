"""English book name to OSIS id mapping.

Bible JSON sources key books by English name, but the index needs a stable id
so citations survive a change of source. The naming differences that actually
bite are Psalm/Psalms, Song of Solomon/Songs/Canticles, and Revelation with or
without "of John", so all of those resolve to the same id here.

Unmapped names raise. Silently dropping a book would leave a Bible index that
looks fine and answers nothing for whatever went missing.
"""

from __future__ import annotations

import re

# Canonical order matters: it drives sort order in citations.
CANON: list[tuple[str, str]] = [
    ("GEN", "Genesis"), ("EXO", "Exodus"), ("LEV", "Leviticus"),
    ("NUM", "Numbers"), ("DEU", "Deuteronomy"), ("JOS", "Joshua"),
    ("JDG", "Judges"), ("RUT", "Ruth"), ("1SA", "1 Samuel"),
    ("2SA", "2 Samuel"), ("1KI", "1 Kings"), ("2KI", "2 Kings"),
    ("1CH", "1 Chronicles"), ("2CH", "2 Chronicles"), ("EZR", "Ezra"),
    ("NEH", "Nehemiah"), ("EST", "Esther"), ("JOB", "Job"),
    ("PSA", "Psalms"), ("PRO", "Proverbs"), ("ECC", "Ecclesiastes"),
    ("SNG", "Song of Solomon"), ("ISA", "Isaiah"), ("JER", "Jeremiah"),
    ("LAM", "Lamentations"), ("EZK", "Ezekiel"), ("DAN", "Daniel"),
    ("HOS", "Hosea"), ("JOL", "Joel"), ("AMO", "Amos"),
    ("OBA", "Obadiah"), ("JON", "Jonah"), ("MIC", "Micah"),
    ("NAM", "Nahum"), ("HAB", "Habakkuk"), ("ZEP", "Zephaniah"),
    ("HAG", "Haggai"), ("ZEC", "Zechariah"), ("MAL", "Malachi"),
    ("MAT", "Matthew"), ("MRK", "Mark"), ("LUK", "Luke"),
    ("JHN", "John"), ("ACT", "Acts"), ("ROM", "Romans"),
    ("1CO", "1 Corinthians"), ("2CO", "2 Corinthians"), ("GAL", "Galatians"),
    ("EPH", "Ephesians"), ("PHP", "Philippians"), ("COL", "Colossians"),
    ("1TH", "1 Thessalonians"), ("2TH", "2 Thessalonians"),
    ("1TI", "1 Timothy"), ("2TI", "2 Timothy"), ("TIT", "Titus"),
    ("PHM", "Philemon"), ("HEB", "Hebrews"), ("JAS", "James"),
    ("1PE", "1 Peter"), ("2PE", "2 Peter"), ("1JN", "1 John"),
    ("2JN", "2 John"), ("3JN", "3 John"), ("JUD", "Jude"),
    ("REV", "Revelation"),
]

OSIS_TO_NAME = dict(CANON)
CANON_INDEX = {osis: i for i, (osis, _) in enumerate(CANON)}

# Alternate spellings and abbreviations seen in the wild.
_ALIASES: dict[str, str] = {
    "psalm": "PSA", "psalms": "PSA", "pss": "PSA",
    "song of songs": "SNG", "songofsongs": "SNG", "canticles": "SNG",
    "song": "SNG", "sos": "SNG",
    "revelation of john": "REV", "revelations": "REV",
    "the revelation": "REV", "apocalypse": "REV",
    "ecclesiastes": "ECC", "qoheleth": "ECC",
    "acts of the apostles": "ACT",
    "1st samuel": "1SA", "2nd samuel": "2SA",
    "1st kings": "1KI", "2nd kings": "2KI",
    "1st chronicles": "1CH", "2nd chronicles": "2CH",
    "1st corinthians": "1CO", "2nd corinthians": "2CO",
    "1st thessalonians": "1TH", "2nd thessalonians": "2TH",
    "1st timothy": "1TI", "2nd timothy": "2TI",
    "1st peter": "1PE", "2nd peter": "2PE",
    "1st john": "1JN", "2nd john": "2JN", "3rd john": "3JN",
    "canticle of canticles": "SNG",
    "philippians": "PHP", "philipians": "PHP",
}

_ORDINALS = {
    "first": "1", "second": "2", "third": "3",
    "i": "1", "ii": "2", "iii": "3",
    "1st": "1", "2nd": "2", "3rd": "3",
}


def _normalize(name: str) -> str:
    text = name.strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"[.\-_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # "First John" / "I John" -> "1 john"
    parts = text.split(" ", 1)
    if len(parts) == 2 and parts[0] in _ORDINALS:
        text = f"{_ORDINALS[parts[0]]} {parts[1]}"

    return text


_LOOKUP: dict[str, str] = {}
for _osis, _name in CANON:
    _LOOKUP[_normalize(_name)] = _osis
    _LOOKUP[_osis.lower()] = _osis
_LOOKUP.update(_ALIASES)
# Singular/plural tolerance for the one book that flips most often.
_LOOKUP["psalm"] = "PSA"


class UnknownBookError(ValueError):
    """Raised when a source book name cannot be mapped to an OSIS id."""


def to_osis(name: str, *, strict: bool = True) -> str | None:
    key = _normalize(name)
    osis = _LOOKUP.get(key)
    if osis:
        return osis

    # Tolerate a trailing/leading "the" and collapsed spacing.
    stripped = key.removeprefix("the ").replace(" ", "")
    for candidate, value in _LOOKUP.items():
        if candidate.replace(" ", "") == stripped:
            return value

    if strict:
        raise UnknownBookError(
            f"Unmapped Bible book name: {name!r}. Add it to lib/bookmap.py "
            f"rather than letting the book be dropped from the index."
        )
    return None


def display_name(osis: str) -> str:
    return OSIS_TO_NAME.get(osis, osis)


def sort_key(osis: str) -> int:
    return CANON_INDEX.get(osis, 999)


def format_ref(osis: str, chapter: int, verse: int, verse_end: int | None = None) -> str:
    base = f"{display_name(osis)} {chapter}:{verse}"
    if verse_end and verse_end != verse:
        base += f"-{verse_end}"
    return base


def _name_variants() -> list[str]:
    """Every spelling a book may appear as in prose.

    The pattern matches only real book names. Matching "any capitalised word
    followed by a number" looks equivalent but is not: in "See 1 John 3:16"
    it consumes "See 1" as the book, discards it as unknown, and then reads
    the remainder as plain John, silently citing the wrong letter.
    """
    ordinals = {"1": ("I", "1st", "First"), "2": ("II", "2nd", "Second"),
                "3": ("III", "3rd", "Third")}
    out: set[str] = set()

    for _, name in CANON:
        out.add(name)
        match = re.match(r"^([123]) (.+)$", name)
        if match:
            number, rest = match.groups()
            out.add(f"{number}{rest}")
            for form in ordinals[number]:
                out.add(f"{form} {rest}")

    out.update({
        "Psalm", "Song of Songs", "Canticles", "Revelation of John",
        "Revelations", "Acts of the Apostles",
    })

    # Longest first so "1 John" wins over "John" at the same position.
    return sorted(out, key=len, reverse=True)


_REF_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(v) for v in _name_variants()) + r")\.?\s*"
    r"(\d{1,3})(?::(\d{1,3})(?:\s*[-\u2013]\s*(\d{1,3}))?)?\b"
)


def extract_refs(text: str) -> list[str]:
    """Pull scripture references out of prose. Used to boost retrieval."""
    found: list[str] = []
    for match in _REF_PATTERN.finditer(text or ""):
        book_raw, chapter, verse, verse_end = match.groups()
        osis = to_osis(book_raw, strict=False)
        if not osis:
            continue
        if verse:
            ref = format_ref(osis, int(chapter), int(verse),
                             int(verse_end) if verse_end else None)
        else:
            ref = f"{display_name(osis)} {chapter}"
        if ref not in found:
            found.append(ref)
    return found


def pack_refs(refs: list[str]) -> str:
    """Store references so book names that contain spaces survive a round trip.

    A space join cannot be split again: "1 John 3:1" becomes four tokens.
    """
    return "\n".join(dict.fromkeys(ref for ref in refs if ref))


def unpack_refs(packed: str) -> list[str]:
    if not packed:
        return []
    if "\n" in packed:
        return [part for part in packed.split("\n") if part]
    # One reference has no delimiter. A space-joined legacy row does not parse
    # as a single reference, so it still splits. Reindex rewrites those rows.
    if parse_ref(packed):
        return [packed]
    return [part for part in packed.split(" ") if part]


_STORED_REF = re.compile(
    r"^(.*)\s+(\d{1,3})(?::(\d{1,3})(?:-(\d{1,3}))?)?$"
)


def parse_ref(ref: str) -> tuple[str, int, int | None, int | None] | None:
    """Return (osis, chapter, verse start, verse end) for a stored reference.

    A chapter reference such as "1 John 3" has no verse bounds.
    """
    match = _STORED_REF.match((ref or "").strip())
    if not match:
        return None
    book, chapter, verse, verse_end = match.groups()
    osis = to_osis(book, strict=False)
    if not osis:
        return None
    start = int(verse) if verse else None
    end = int(verse_end) if verse_end else start
    return osis, int(chapter), start, end


def refs_overlap(left: str, right: str) -> bool:
    """True when two references name the same book and chapter and the verses meet.

    A chapter reference overlaps every verse in that chapter.
    """
    a = parse_ref(left)
    b = parse_ref(right)
    if not a or not b:
        return False
    if a[0] != b[0] or a[1] != b[1]:
        return False
    if a[2] is None or b[2] is None:
        return True
    a_end = a[3] if a[3] is not None else a[2]
    b_end = b[3] if b[3] is not None else b[2]
    return a[2] <= b_end and b[2] <= a_end


def any_overlap(left, right) -> bool:
    rights = list(right)
    return any(refs_overlap(a, b) for a in left for b in rights)
