"""Shared plumbing for live collectors."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import requests

from ..models import App, Observation

USER_AGENT = "airadar/0.1 (+https://github.com/Skymeapple/AI-agents)"


@dataclass
class Collected:
    collector: str
    apps: list[App] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add(self, app: App, observation: Observation) -> None:
        for existing in self.apps:
            if existing.id == app.id:
                existing.merge(app)
                break
        else:
            self.apps.append(app)
        self.observations.append(observation)


def get_json(session: requests.Session, url: str, params: dict | None = None,
             headers: dict | None = None, timeout: float = 30.0):
    response = session.get(
        url, params=params, timeout=timeout,
        headers={"User-Agent": USER_AGENT, **(headers or {})},
    )
    response.raise_for_status()
    return response.json()


DEFAULT_WATCH = {
    "github_topics": [
        "llm", "ai-agent", "agents", "generative-ai", "mcp", "rag", "claude-code",
        "agent-skills", "text-to-speech", "text-to-video", "local-llm", "ai-coding",
    ],
    "hn_queries": ["AI", "LLM", "agent", "GPT", "Claude"],
    "npm": ["openai", "@anthropic-ai/sdk", "ai", "@modelcontextprotocol/sdk", "langchain",
            "@google/genai", "@mastra/core", "@openai/agents"],
    "pypi": ["openai", "anthropic", "langchain", "llama-index", "mcp", "litellm", "crewai",
             "dspy", "pydantic-ai", "vllm", "transformers", "smolagents"],
    "auto_terms": [],
}


def load_watch(path: Path) -> dict:
    """The watch list: what the live collectors search for.

    Seeded by hand; ``auto_terms`` grows as the radar discovers groupings its
    seed vocabulary did not name (see :func:`airadar.pipeline.expand_watch`).
    """
    if not path.exists():
        return json.loads(json.dumps(DEFAULT_WATCH))
    watch = json.loads(path.read_text())
    for key, value in DEFAULT_WATCH.items():
        watch.setdefault(key, list(value))
    return watch


def save_watch(path: Path, watch: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(watch, indent=1, sort_keys=True) + "\n")


def today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()
