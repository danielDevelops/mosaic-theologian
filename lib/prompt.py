"""Prompt construction and the source hierarchy.

The rules below are the whole point of the project: answer from Scripture
first, then what this church actually teaches, and be explicit when the
church has not addressed something rather than inventing a position for it.
"""

from __future__ import annotations

from .retrieval import RetrievalResult

SYSTEM_PROMPT = """\
You are a study assistant for one specific local church. You answer from the \
sources provided in each request, and you are careful about whose voice is \
whose.

Order of authority, highest first:

1. SCRIPTURE - the biblical text supplied below.
2. CHURCH BELIEFS - this church's own doctrinal statements.
3. CHURCH TEACHING - sermons and articles published by this church.
4. GENERAL KNOWLEDGE - your own theological and historical knowledge.

Rules:

- Ground the answer in the supplied passages. Quote or reference them by their \
citation labels so every claim can be checked.
- Where Scripture speaks to the question, lead with it, then weave in this \
church's belief statements and sermons in the same explanation.
- Never attribute a position to this church unless the supplied CHURCH BELIEFS \
or CHURCH TEACHING passages actually support it. If they do not address the \
question, say so in the same answer.
- You may use general knowledge to define terms, explain historical debates, \
and give context the church never addressed directly. When a point is only \
general knowledge and is not in the supplied sources, mark that sentence in \
place so it is not attributed to the church. Do not give it a heading of its own.
- When the sources genuinely conflict or a question is historically contested, \
present the positions fairly inside the same answer.
- A passage marked "Connected in" a sermon is a link that sermon drew. Attribute \
the link to the sermon. Do not treat it as one biblical text citing the other, \
and do not treat it as what the sermon's main passage itself says.
- A passage marked "Also teaches" is another sermon on that same passage. \
Attribute it to that sermon.
- Be direct and pastoral. Do not pad. If you do not know, say so.
- Write one integrated answer. Do not use section headings such as "what the \
Bible says", "what Mosaic says", or "wider Christian thought".
- Do not write a sources list. Citations are printed after your reply.
"""


def _render_block(title: str, passages, note: str = "") -> str:
    if not passages:
        return f"### {title}\n(none retrieved)\n"

    lines = [f"### {title}"]
    if note:
        lines.append(f"_{note}_")
    for passage in passages:
        lines.append(f"\n[{passage.label}]")
        if getattr(passage, "connection", ""):
            lines.append(passage.connection)
        lines.append(passage.text.strip())
    return "\n".join(lines) + "\n"


def build_user_message(result: RetrievalResult) -> str:
    scripture = [p for p in result.by_collection("scripture") if not p.connection]
    beliefs = result.by_collection("beliefs")
    mosaic = [p for p in result.by_collection("mosaic") if not p.connection]
    linked = [p for p in result.passages if p.connection]

    teaching_note = "Each label is title | speaker | date."
    if any(p.connection.startswith("Connected in") for p in linked):
        teaching_note += (
            " A line that begins Connected in names the sermon that drew that "
            "link. It is not a claim that one biblical text cites the other."
        )
    if any(p.connection.startswith("Also teaches") for p in linked):
        teaching_note += (
            " A line that begins Also teaches names the passage that joins "
            "that sermon to one already retrieved."
        )

    parts = [
        "Answer the question using the sources below.",
        "",
        _render_block("SCRIPTURE", scripture),
        _render_block("CHURCH BELIEFS", beliefs),
        _render_block(
            "CHURCH TEACHING (sermons and articles)", mosaic + linked,
            note=teaching_note,
        ),
    ]

    if result.mosaic_is_silent:
        parts.append(
            "NOTE: retrieval found nothing closely relevant from this church's "
            "own material. Say clearly that the church has not addressed this "
            "directly, then answer from Scripture and general knowledge.\n"
        )

    parts.append(f"QUESTION: {result.question}")
    return "\n".join(parts)


def build_messages(result: RetrievalResult,
                   history: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-6:]:
        messages.append(turn)
    messages.append({"role": "user", "content": build_user_message(result)})
    return messages


def format_citations(result: RetrievalResult) -> list[str]:
    seen: list[str] = []
    for passage in result.passages:
        label = passage.label
        if passage.connection:
            label = f"{label} ({passage.connection})"
        if passage.url and passage.collection != "scripture":
            label = f"{label} — {passage.url}"
        if label not in seen:
            seen.append(label)
    return seen
