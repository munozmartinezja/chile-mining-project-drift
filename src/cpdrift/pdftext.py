"""Cached access to PDF text, words, tables, and page geometry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pdfplumber

CACHE_DIR = Path("data/interim/cache")


def _cache_path(path: Path) -> Path:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return CACHE_DIR / f"{digest.hexdigest()[:16]}.json"


def _extract(path: Path) -> dict[str, Any]:
    with pdfplumber.open(path) as document:
        pages = [
            {
                "text": page.extract_text(x_tolerance=2, y_tolerance=3) or "",
                "words": page.extract_words(x_tolerance=1, y_tolerance=2),
                "tables": page.extract_tables(),
                "width": float(page.width),
                "height": float(page.height),
            }
            for page in document.pages
        ]
    return {"pages": pages}


def get_pdf_data(path: Path) -> dict[str, Any]:
    """Return all cached page representations, extracting them together on a miss."""
    cache_path = _cache_path(path)
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    data = _extract(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(data, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    return data


def get_pages_text(path: Path) -> list[str]:
    """Return one extracted string per page."""
    return [str(page["text"]) for page in get_pdf_data(path)["pages"]]


def get_pages_words(path: Path) -> list[list[dict[str, Any]]]:
    """Return one coordinate-bearing word list per page."""
    return [list(page["words"]) for page in get_pdf_data(path)["pages"]]
