"""Verify declared edition controls against curated reference citations."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .pdftext import get_pages_text

REFERENCE_PATH = Path("data/reference/totales_declarados.csv")
REFERENCE_COLUMNS = [
    "edicion",
    "n_declarado",
    "pagina_n",
    "frase_n",
    "total_mmusd",
    "pagina_total",
    "frase_total",
    "umbral_mmusd",
    "pagina_umbral",
    "frase_umbral",
]
CITATIONS = (
    ("proyectos_declarados", "n_declarado", "pagina_n", "frase_n", "número"),
    ("total_declarado_mmusd", "total_mmusd", "pagina_total", "frase_total", "total"),
    ("umbral_edicion", "umbral_mmusd", "pagina_umbral", "frase_umbral", "umbral"),
)


def normalize_citation(text: str) -> str:
    """Normalize citation text without changing digits or punctuation."""
    decomposed = unicodedata.normalize("NFD", text.casefold())
    without_accents = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_accents).strip()


def citation_in_pages(
    pages: list[str], phrase: str, page_number: int
) -> tuple[bool, list[int]]:
    """Search a normalized citation on its page and adjacent pages."""
    searched = [
        candidate
        for candidate in (page_number - 1, page_number, page_number + 1)
        if 1 <= candidate <= len(pages)
    ]
    expected = normalize_citation(phrase)
    found = any(
        expected in normalize_citation(pages[item - 1]) for item in searched
    )
    return found, searched


def _optional_value(value: object) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def _optional_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def load_references(path: Path = REFERENCE_PATH) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    if frame.columns.tolist() != REFERENCE_COLUMNS:
        raise ValueError(f"columnas inválidas en referencia de totales: {path}")
    return frame.to_dict("records")


def _citation_finding(
    pages: list[str] | None,
    value: object,
    page_value: object,
    phrase_value: object,
    label: str,
) -> dict[str, Any] | None:
    phrase = _optional_text(phrase_value)
    if not phrase:
        return None
    page = _optional_value(page_value)
    if page is None:
        raise ValueError(f"cita de {label} sin página")
    page_number = int(page)
    if pages is None:
        status = "sin_pdf"
        searched: list[int] = []
    else:
        found, searched = citation_in_pages(pages, phrase, page_number)
        status = "verificada" if found else "no_encontrada"
    return {
        "valor": _optional_value(value),
        "cita": phrase,
        "pagina": page_number,
        "estado": status,
        "paginas_buscadas": searched,
        "tipo": label,
    }


def verify_reference(
    pdf_path: Path, reference: Mapping[str, object]
) -> dict[str, Any]:
    pages = get_pages_text(pdf_path) if pdf_path.exists() else None
    result: dict[str, Any] = {"edicion": int(reference["edicion"])}
    for result_key, value_key, page_key, phrase_key, label in CITATIONS:
        result[result_key] = _citation_finding(
            pages,
            reference.get(value_key),
            reference.get(page_key),
            reference.get(phrase_key),
            label,
        )
    return result


def raise_for_unverified(finding: Mapping[str, Any]) -> None:
    failures = [
        citation
        for key, *_ in CITATIONS
        if (citation := finding.get(key))
        and citation["estado"] == "no_encontrada"
    ]
    if not failures:
        return
    edition = int(finding["edicion"])
    detail = "; ".join(
        f"{citation['tipo']} página {citation['pagina']}: no_encontrada"
        for citation in failures
    )
    raise ValueError(f"Referencia declarada {edition}: {detail}")


def extract_declared(
    pdf_path: Path,
    edition: int,
    reference_path: Path = REFERENCE_PATH,
) -> dict[str, Any]:
    references = load_references(reference_path)
    reference = next(
        (item for item in references if int(item["edicion"]) == edition),
        None,
    )
    if reference is None:
        raise ValueError(f"edición {edition} ausente en referencia de totales")
    finding = verify_reference(pdf_path, reference)
    raise_for_unverified(finding)
    return finding


def _value(finding: dict[str, Any] | None) -> Any:
    return finding["valor"] if finding else None


def reconcile(frame: pd.DataFrame, finding: dict[str, Any]) -> dict[str, Any]:
    edition = int(finding["edicion"])
    rows = frame.loc[frame["edicion"].eq(edition)]
    row_count = len(rows)
    raw_sum = rows["inversion_mmusd"].sum(min_count=1) if row_count else 0.0
    extracted_sum = None if pd.isna(raw_sum) else float(raw_sum)
    declared_count = _value(finding["proyectos_declarados"])
    declared_total = _value(finding["total_declarado_mmusd"])
    rows_ok = declared_count is None or abs(row_count - int(declared_count)) <= 3
    sum_ok = (
        declared_total is not None
        and declared_total != 0
        and extracted_sum is not None
        and abs(extracted_sum - declared_total) / declared_total <= 0.005
    )
    status = "ok" if rows_ok and sum_ok else "revisar"
    return {
        "edicion": edition,
        "filas": row_count,
        "declarados": declared_count,
        "suma_mmusd": round(extracted_sum, 3) if extracted_sum is not None else None,
        "total_mmusd": declared_total,
        "estado": status,
        "umbral_edicion": _value(finding["umbral_edicion"]),
        "citas": {
            "proyectos": finding["proyectos_declarados"],
            "total": finding["total_declarado_mmusd"],
            "umbral": finding["umbral_edicion"],
        },
    }


def extract_declared_all(
    raw_dir: Path,
    extracted: pd.DataFrame,
    reference_path: Path = REFERENCE_PATH,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for reference in load_references(reference_path):
        edition = int(reference["edicion"])
        finding = verify_reference(raw_dir / f"cartera_{edition}.pdf", reference)
        raise_for_unverified(finding)
        results.append(reconcile(extracted, finding))
    return results
