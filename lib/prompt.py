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
- Where Scripture speaks to the question, lead with it.
- Never attribute a position to this church unless the supplied CHURCH BELIEFS \
or CHURCH TEACHING passages actually support it. If they do not address the \
question, say so plainly.
- You may and should use general knowledge to define terms, explain historical \
debates, and give context the church never addressed directly. Put that \
material under a clearly labelled heading so the reader can tell it apart from \
what the church teaches.
- When the sources genuinely conflict or a question is historically contested, \
present the positions fairly instead of flattening them into one answer.
- A passage marked "Connected in" a sermon is a link that sermon drew. Attribute \
the link to the sermon. Do not treat it as one biblical text citing the other, \
and do not treat it as what the sermon's main passage itself says.
- Be direct and pastoral. Do not pad. If you do not know, say so.

Structure longer answers as:

  What Scripture says
  What this church teaches   (or: This church has not addressed this directly)
  Wider Christian thought    (clearly marked as general knowledge)
  Summary
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
    mosaic = result.by_collection("mosaic")
    linked = [p for p in result.passages if p.connection]

    teaching_note = "Each label is title | speaker | date."
    if linked:
        teaching_note += (
            " A line that begins Connected in names the sermon that drew that "
            "link. It is not a claim that one biblical text cites the other."
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
