from pathlib import Path
from typing import Self

from cpdrift import pdftext


class SyntheticPage:
    width = 612
    height = 792

    def extract_text(self, **_: object) -> str:
        return "Página sintética"

    def extract_words(self, **_: object) -> list[dict[str, object]]:
        return [{"text": "Página", "x0": 10, "top": 20}]

    def extract_tables(self) -> list[list[list[str]]]:
        return [[['celda']]]


class SyntheticDocument:
    def __init__(self) -> None:
        self.pages = [SyntheticPage()]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def test_second_cached_read_does_not_open_pdf(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "synthetic.pdf"
    path.write_bytes(b"%PDF synthetic fixture")
    calls = 0

    def counted_open(_: Path) -> SyntheticDocument:
        nonlocal calls
        calls += 1
        return SyntheticDocument()

    monkeypatch.setattr(pdftext, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(pdftext.pdfplumber, "open", counted_open)

    assert pdftext.get_pages_text(path) == ["Página sintética"]
    assert pdftext.get_pages_words(path) == [[{"text": "Página", "x0": 10, "top": 20}]]
    assert pdftext.get_pages_text(path) == ["Página sintética"]
    assert calls == 1
