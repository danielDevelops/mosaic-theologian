"""HTML to structured text for Mosaic pages.

Extracts the readable body, heading sections, and message metadata (title,
date, speaker, campus, audio URL, cited references). Kept deliberately
tolerant: the archive spans 2006 to today and older pages do not follow the
current template.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse

from bs4 import BeautifulSoup

from . import bookmap

_STRIP_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "form", "svg")

_DATE_PATTERNS = [
    re.compile(r"\b(January|February|March|April|May|June|July|August|September|"
               r"October|November|December)\s+(\d{1,2}),\s*(\d{4})\b"),
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
]

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}

# "WG: Renaut van der Riet", "WDW: Danny Conner"
_SPEAKER_CAMPUS = re.compile(
    r"\b(WG|WDW|ESP|ES)\s*:\s*([A-Z][A-Za-z.'\-]+(?:\s+[A-Za-z.'\-]+){0,3})"
)
_BARE_SPEAKER = re.compile(
    r"\b((?:Renaut van der Riet|Brady White|Joel Coffman|Danny Conner|"
    r"Zack Olsen|Caleb Carine|Kevin Richardson))\b"
)

_AUDIO_SUFFIX_CAMPUS = re.compile(r"(\d{8})([a-z]{2,4})\.mp3", re.IGNORECASE)


@dataclass
class ExtractedPage:
    url: str
    title: str = ""
    body: str = ""
    sections: list[tuple[str, str]] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    audio_url: str = ""
    date: str = ""
    speaker: str = ""
    campus: str = ""
    series: str = ""
    scripture_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "body": self.body,
            "sections": [list(s) for s in self.sections],
            "audio_url": self.audio_url,
            "date": self.date,
            "speaker": self.speaker,
            "campus": self.campus,
            "series": self.series,
            "scripture_refs": self.scripture_refs,
        }


def _normalize_date(text: str) -> str:
    for pattern in _DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groups()
        try:
            if groups[0].lower() in _MONTHS:
                month = _MONTHS[groups[0].lower()]
                return datetime(int(groups[2]), month, int(groups[1])).strftime("%Y-%m-%d")
            return datetime(int(groups[0]), int(groups[1]), int(groups[2])).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


def _pretty_slug(slug: str) -> str:
    """Title-case a URL slug without mangling ordinals ("1st", not "1St")."""
    words = []
    for word in slug.replace("-", " ").split():
        if re.fullmatch(r"\d+(st|nd|rd|th)", word, re.IGNORECASE):
            words.append(word.lower())
        else:
            words.append(word[:1].upper() + word[1:])
    return " ".join(words)


def series_from_url(url: str) -> str:
    parts = [p for p in urlparse(url).path.strip("/").split("/") if p]
    if len(parts) >= 2 and parts[0] == "messages":
        return _pretty_slug(parts[1])
    return ""


def is_message_url(url: str) -> bool:
    """A message page is /messages/<series>/<slug>/, not a listing."""
    parts = [p for p in urlparse(url).path.strip("/").split("/") if p]
    if not parts or parts[0] != "messages":
        return False
    if len(parts) < 3:
        return False
    return parts[1] not in {"archive", "series"}


def canonical_url(url: str) -> str:
    """Collapse URL variants that address the same page.

    Series pages are linked three ways: bare, ?location=wg, and
    ?location=wdw. They are one page under a filter, not three pages, and the
    links we harvest are the same. Without collapsing them the crawler pays a
    separate HTTP fetch for each variant and then throws two away, which turns
    103 series pages into 309 requests before it reaches a single sermon.

    Identity is the path. Query keys that only filter a view are dropped;
    anything else is preserved so genuinely distinct pages stay distinct.
    """
    parsed = urlparse(url.split("#")[0])
    view_only = {"location", "_", "share", "replytocom", "fbclid", "utm_source",
                 "utm_medium", "utm_campaign"}

    kept = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=False)
            if k.lower() not in view_only]

    path = parsed.path.rstrip("/") or "/"
    rebuilt = f"{parsed.scheme}://{parsed.netloc.lower()}{path}"
    if kept:
        rebuilt += "?" + urlencode(sorted(kept))
    return rebuilt


def is_listing_url(url: str) -> bool:
    """True for index pages that gain links when a sermon is published.

    The archive, each series index, and the site root all change over time,
    so they have to be re-fetched to discover new messages. Message detail
    pages do not change once published and are never re-fetched.
    """
    path = urlparse(url).path.rstrip("/")
    if path in ("", "/"):
        return True
    parts = [p for p in path.strip("/").split("/") if p]
    if not parts or parts[0] != "messages":
        return False
    return not is_message_url(url)


def campus_from_audio(audio_url: str) -> str:
    match = _AUDIO_SUFFIX_CAMPUS.search(audio_url or "")
    return match.group(2).lower() if match else ""


def extract(html: str, url: str, audio_hosts: list[str] | None = None) -> ExtractedPage:
    soup = BeautifulSoup(html, "lxml")
    page = ExtractedPage(url=url)

    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()

    heading = soup.find("h1")
    if heading:
        page.title = heading.get_text(" ", strip=True)
    elif soup.title:
        page.title = soup.title.get_text(" ", strip=True)
    page.title = re.sub(r"\s*[-|]\s*Mosaic Church\s*$", "", page.title).strip()

    # Links, absolute and same-origin filtered later by the crawler.
    for anchor in soup.find_all("a", href=True):
        href = urljoin(url, anchor["href"].strip())
        page.links.append(href.split("#")[0])

    # Audio: an <audio src>, a direct .mp3 link, or a bare URL in the text.
    hosts = audio_hosts or []
    candidates: list[str] = []
    for node in soup.find_all(["audio", "source"]):
        src = node.get("src")
        if src:
            candidates.append(urljoin(url, src))
    for anchor in soup.find_all("a", href=True):
        if ".mp3" in anchor["href"].lower():
            candidates.append(urljoin(url, anchor["href"]))

    raw_text = soup.get_text("\n", strip=True)
    for match in re.finditer(r"https?://[^\s\"'<>]+\.mp3", raw_text, re.IGNORECASE):
        candidates.append(match.group(0))

    for candidate in candidates:
        host = urlparse(candidate).netloc.lower()
        if not hosts or any(h in host for h in hosts) or candidate.lower().endswith(".mp3"):
            page.audio_url = candidate
            break

    # Heading sections. Belief statements and article bodies both use these.
    sections: list[tuple[str, str]] = []
    current_heading = ""
    buffer: list[str] = []

    for node in soup.find_all(["h2", "h3", "h4", "p", "li", "dt", "dd", "blockquote"]):
        text = node.get_text(" ", strip=True)
        if not text:
            continue
        if node.name in ("h2", "h3", "h4"):
            if buffer:
                sections.append((current_heading, "\n".join(buffer).strip()))
                buffer = []
            current_heading = text
        else:
            buffer.append(text)

    if buffer:
        sections.append((current_heading, "\n".join(buffer).strip()))

    page.sections = [(h, b) for h, b in sections if b]
    page.body = "\n\n".join(
        (f"{h}\n{b}" if h else b) for h, b in page.sections
    ).strip()

    page.date = _normalize_date(raw_text[:4000])

    speaker_match = _SPEAKER_CAMPUS.search(raw_text)
    if speaker_match:
        page.campus = speaker_match.group(1).lower()
        page.speaker = speaker_match.group(2).strip()
    else:
        bare = _BARE_SPEAKER.search(raw_text)
        if bare:
            page.speaker = bare.group(1)

    if not page.campus:
        page.campus = campus_from_audio(page.audio_url) or "wg"

    page.series = series_from_url(url)
    page.scripture_refs = bookmap.extract_refs(page.body)

    return page


def normalize_audio_url(audio_url: str) -> str:
    """Drop the cache-busting query so the same file yields one stable key.

    The site appends "?_=1" to sermon audio. Leaving it in would make the key
    depend on a value that has nothing to do with which sermon this is.
    """
    if not audio_url:
        return ""
    parsed = urlparse(audio_url.strip())
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".lower()


def dedup_key(date: str, campus: str, audio_url: str) -> str:
    """Identity for a sermon.

    Page URL is deliberately not part of this. The same message is published
    under more than one series path, and keying on URL would index it twice
    and let one sermon dominate retrieval.
    """
    if audio_url:
        return f"audio:{normalize_audio_url(audio_url)}"
    return f"msg:{date}|{campus}".lower()
