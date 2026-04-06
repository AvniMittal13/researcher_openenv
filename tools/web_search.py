"""Web search tool using DuckDuckGo."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Return up to *max_results* web results for *query*.

    Each result is ``{"title": ..., "url": ..., "snippet": ...}``.
    Returns an empty list on failure rather than raising.
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        return [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in raw
        ]
    except Exception as exc:
        logger.warning("web_search failed for %r: %s", query, exc)
        return []
