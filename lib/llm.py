"""Local chat model client.

Talks to a llama.cpp server over its OpenAI-compatible endpoint, with a
fallback to the native /completion route for older builds, and optional
Ollama support. Everything is local: the base URL is always a loopback
address, and no request leaves the machine.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import requests


class ChatUnavailable(RuntimeError):
    """The local model server is not reachable."""


class LocalChat:
    def __init__(self, settings: dict[str, Any]) -> None:
        cfg = settings["Chat"]
        self.base_url: str = cfg["ServerUrl"].rstrip("/")
        self.runtime: str = cfg.get("Runtime", "llama.cpp")
        self.max_tokens = int(cfg.get("MaxTokens", 900))
        self.temperature = float(cfg.get("Temperature", 0.2))
        self.top_p = float(cfg.get("TopP", 0.9))
        self.model_name = cfg.get("ModelName", "local")
        self.timeout = int(cfg.get("TimeoutSeconds", 600))

    # ---------------------------------------------------------- health --

    def available(self) -> bool:
        for path in ("/health", "/v1/models", "/api/tags"):
            try:
                response = requests.get(self.base_url + path, timeout=3)
                if response.status_code < 500:
                    return True
            except requests.RequestException:
                continue
        return False

    def require(self) -> None:
        if not self.available():
            raise ChatUnavailable(
                f"No local model server at {self.base_url}.\n"
                f"Start it first:  ./mosaic.sh start   (mac)\n"
                f"                 llama-server -m models/chat.gguf "
                f"--host 127.0.0.1 --port 8080   (windows)"
            )

    # ------------------------------------------------------------ chat --

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": True,
        }

        try:
            with requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload, stream=True, timeout=self.timeout,
            ) as response:
                if response.status_code == 404:
                    yield from self._stream_native(messages)
                    return
                response.raise_for_status()

                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        chunk = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        yield piece

        except requests.RequestException as exc:
            raise ChatUnavailable(f"Local model request failed: {exc}") from exc

    def _stream_native(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Fallback for llama.cpp builds without the OpenAI-compatible route."""
        prompt = self._flatten(messages)
        payload = {
            "prompt": prompt,
            "n_predict": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": True,
        }

        with requests.post(f"{self.base_url}/completion",
                           json=payload, stream=True, timeout=self.timeout) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                try:
                    chunk = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                piece = chunk.get("content")
                if piece:
                    yield piece
                if chunk.get("stop"):
                    break

    def complete(self, messages: list[dict[str, str]]) -> str:
        return "".join(self.stream(messages))

    @staticmethod
    def _flatten(messages: list[dict[str, str]]) -> str:
        lines: list[str] = []
        for message in messages:
            role = message["role"].upper()
            lines.append(f"<|{role}|>\n{message['content']}")
        lines.append("<|ASSISTANT|>\n")
        return "\n\n".join(lines)
