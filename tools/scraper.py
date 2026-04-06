"""URL scraper with HTML and PDF support."""

from __future__ import annotations

import logging
import re
from typing import Any

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_TIMEOUT = 30
_MAX_TEXT = 8000
_MAX_PDF_PAGES = 12
_HEADERS = {"User-Agent": "research-agent/1.0"}
_ARXIV_ABS_RE = re.compile(r"arxiv\.org/abs/([\d.]+)")


def scrape(url: str) -> dict[str, Any]:
    """Fetch *url* and return ``{"type": "html"|"pdf", "content": str}``.

    Automatically follows the ar5iv HTML path for arxiv abstract pages.
    Returns an error dict on failure.
    """
    try:
        # arxiv abstract → try ar5iv HTML first (much cleaner)
        m = _ARXIV_ABS_RE.search(url)
        if m:
            ar5iv_url = f"https://ar5iv.org/html/{m.group(1)}"
            try:
                return _scrape_html(ar5iv_url)
            except Exception:
                pass  # fall through to original URL

        resp = requests.get(url, timeout=_TIMEOUT, headers=_HEADERS)
        resp.raise_for_status()

        ct = resp.headers.get("Content-Type", "")
        if "application/pdf" in ct or url.rstrip("/").endswith(".pdf"):
            return _parse_pdf(resp.content)

        return _parse_html(resp.text)

    except Exception as exc:
        logger.warning("scrape failed for %r: %s", url, exc)
        return {"type": "error", "content": f"Scrape failed: {exc}"}


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _scrape_html(url: str) -> dict[str, Any]:
    resp = requests.get(url, timeout=_TIMEOUT, headers=_HEADERS)
    resp.raise_for_status()
    return _parse_html(resp.text)


def _parse_html(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return {"type": "html", "content": text[:_MAX_TEXT]}


def _parse_pdf(content: bytes) -> dict[str, Any]:
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        for i, page in enumerate(doc):
            if i >= _MAX_PDF_PAGES:
                break
            text += page.get_text()
        doc.close()
        return {"type": "pdf", "content": text[:_MAX_TEXT]}
    except ImportError:
        return {"type": "error", "content": "PyMuPDF not installed — cannot parse PDF"}
    except Exception as exc:
        return {"type": "error", "content": f"PDF parse failed: {exc}"}
