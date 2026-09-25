"""Interactive chat over the local index.

Type any question. Nothing is pre-seeded: the question is embedded at ask
time and matched against stored source text, so the set of answerable
questions is not fixed in advance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import Paths, apply_offline_env, load_settings  # noqa: E402
from lib.llm import ChatUnavailable, LocalChat                   # noqa: E402
from lib.prompt import build_messages, format_citations          # noqa: E402
from lib.retrieval import Retriever, expansion_phrases           # noqa: E402

BANNER = """\
Mosaic study assistant  (local, offline)
  Ask anything. Commands: :sources  :clear  :quit
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the local index a question")
    parser.add_argument("question", nargs="*", help="Ask once and exit")
    parser.add_argument("--device", default=None,
                        help="Embedding device override, e.g. cpu")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the embedding identity check (not advised)")
    parser.add_argument("--show-context", action="store_true",
                        help="Print the retrieved passages before answering")
    args = parser.parse_args()

    apply_offline_env()
    settings = load_settings()
    paths = Paths()

    if not (paths.index / "index-manifest.json").is_file():
        print("No index found. Build it on the Windows machine first, or run "
              "./mosaic.sh verify to check a copied bundle.")
        return 1

    try:
        retriever = Retriever(settings, paths.index,
                              device=args.device, verify=not args.no_verify)
    except RuntimeError as exc:
        print(f"\n{exc}\n")
        return 2

    chat = LocalChat(settings)
    try:
        chat.require()
    except ChatUnavailable as exc:
        print(f"\n{exc}\n")
        return 3

    counts = retriever.store.counts()
    history: list[dict[str, str]] = []
    last_result = None

    one_shot = " ".join(args.question).strip()
    if not one_shot:
        print(BANNER)
        print("  index: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))
        print()

    while True:
        if one_shot:
            question = one_shot
        else:
            try:
                question = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0

        if not question:
            continue

        lowered = question.lower()
        if lowered in (":quit", ":q", "exit", "quit"):
            return 0
        if lowered == ":clear":
            history.clear()
            print("History cleared.\n")
            continue
        if lowered == ":sources":
            if last_result:
                for citation in format_citations(last_result):
                    print(f"  - {citation}")
            else:
                print("  (nothing retrieved yet)")
            print()
            continue

        try:
            phrases = expansion_phrases(
                chat, question, retriever.expansion_max_queries,
            )
            result = retriever.search(question, phrases)
        except ChatUnavailable as exc:
            print(f"\n{exc}")
            return 3
        last_result = result

        if args.show_context:
            print()
            for passage in result.passages:
                print(f"  [{passage.collection}] {passage.label}  "
                      f"score={passage.score:.3f}")
            print()

        messages = build_messages(result, history)

        print()
        answer_parts: list[str] = []
        try:
            for piece in chat.stream(messages):
                sys.stdout.write(piece)
                sys.stdout.flush()
                answer_parts.append(piece)
        except ChatUnavailable as exc:
            print(f"\n{exc}")
            return 3
        except KeyboardInterrupt:
            print("\n[interrupted]")

        answer = "".join(answer_parts)
        print("\n")

        citations = format_citations(result)
        if citations:
            print("Sources")
            for citation in citations:
                print(f"  - {citation}")
            print()

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        if one_shot:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
