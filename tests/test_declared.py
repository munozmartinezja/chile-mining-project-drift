import json
from pathlib import Path

import pandas as pd
import pytest

from cpdrift import declared


def reference_row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "edicion": 2024,
        "n_declarado": 51,
        "pagina_n": 2,
        "frase_n": "abarca un total de 51 proyectos",
        "total_mmusd": 100,
        "pagina_total": 2,
        "frase_total": "inversión estimada de 100 millones de dólares",
        "umbral_mmusd": 30,
        "pagina_umbral": 2,
        "frase_umbral": "inversiones superiores a 30 millones de dólares",
    }
    row.update(changes)
    return row


def test_reference_phrase_on_next_page_is_verified(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "cartera_2024.pdf"
    pdf_path.write_bytes(b"")
    monkeypatch.setattr(
        declared,
        "get_pages_text",
        lambda _: ["", "", "La INVERSION   estimada de 100 millones de dolares."],
    )

    result = declared.verify_reference(
        pdf_path,
        reference_row(
            n_declarado=None,
            frase_n="",
            umbral_mmusd=None,
            frase_umbral="",
        ),
    )

    assert result["total_declarado_mmusd"]["estado"] == "verificada"
    assert result["total_declarado_mmusd"]["paginas_buscadas"] == [1, 2, 3]


def test_missing_reference_phrase_reports_status_and_raises(
    tmp_path: Path, monkeypatch
) -> None:
    pdf_path = tmp_path / "cartera_2024.pdf"
    pdf_path.write_bytes(b"")
    monkeypatch.setattr(declared, "get_pages_text", lambda _: ["", "texto diferente", ""])

    result = declared.verify_reference(
        pdf_path,
        reference_row(n_declarado=None, frase_n="", umbral_mmusd=None, frase_umbral=""),
    )

    assert result["total_declarado_mmusd"]["estado"] == "no_encontrada"
    assert result["total_declarado_mmusd"]["paginas_buscadas"] == [1, 2, 3]
    with pytest.raises(ValueError, match=r"2024.*total.*página 2.*no_encontrada"):
        declared.raise_for_unverified(result)


@pytest.mark.parametrize(
    ("page_text", "reference_text"),
    [
        (
            "inversión estimada de 10² millones de dólares",
            "inversión estimada de 102 millones de dólares",
        ),
        (
            "inversión estimada… de 102 millones de dólares",
            "inversión estimada... de 102 millones de dólares",
        ),
    ],
)
def test_reference_comparison_preserves_digits_and_punctuation(
    tmp_path: Path, monkeypatch, page_text: str, reference_text: str
) -> None:
    pdf_path = tmp_path / "cartera_2024.pdf"
    pdf_path.write_bytes(b"")
    monkeypatch.setattr(declared, "get_pages_text", lambda _: [page_text])

    result = declared.verify_reference(
        pdf_path,
        reference_row(
            n_declarado=None,
            frase_n="",
            pagina_total=1,
            frase_total=reference_text,
            umbral_mmusd=None,
            frase_umbral="",
        ),
    )

    assert result["total_declarado_mmusd"]["estado"] == "no_encontrada"


def test_edition_without_pdf_marks_citations_without_error(tmp_path: Path) -> None:
    result = declared.verify_reference(tmp_path / "cartera_2025.pdf", reference_row())

    assert result["proyectos_declarados"]["estado"] == "sin_pdf"
    assert result["total_declarado_mmusd"]["estado"] == "sin_pdf"
    assert result["umbral_edicion"]["estado"] == "sin_pdf"
    declared.raise_for_unverified(result)


@pytest.mark.parametrize(
    ("row_count", "extracted_sum", "expected"),
    [
        (50, 100.4, "ok"),
        (47, 100.0, "revisar"),
        (51, 100.6, "revisar"),
    ],
)
def test_reconciliation_uses_reference_tolerances(
    row_count: int, extracted_sum: float, expected: str
) -> None:
    frame = pd.DataFrame(
        {
            "edicion": [2024] * row_count,
            "inversion_mmusd": [extracted_sum / row_count] * row_count,
        }
    )
    finding = declared.verify_reference(Path("missing.pdf"), reference_row())

    result = declared.reconcile(frame, finding)

    assert result["estado"] == expected


def test_reconciliation_without_declared_count_uses_only_sum() -> None:
    frame = pd.DataFrame({"edicion": [2025], "inversion_mmusd": [104549.0]})
    finding = declared.verify_reference(
        Path("missing.pdf"),
        reference_row(
            edicion=2025,
            n_declarado=None,
            pagina_n=None,
            frase_n="",
            total_mmusd=104549,
        ),
    )

    assert declared.reconcile(frame, finding)["estado"] == "ok"


def test_2022_reconciliation_is_without_rows() -> None:
    frame = pd.DataFrame(columns=["edicion", "inversion_mmusd"])
    finding = declared.verify_reference(
        Path("missing.pdf"), reference_row(edicion=2022, n_declarado=53)
    )

    assert declared.reconcile(frame, finding)["estado"] == "sin_filas"


@pytest.mark.parametrize(
    "investments",
    [
        pd.Series([float("nan")], dtype=float),
        pd.Series([pd.NA], dtype="Float64"),
    ],
)
def test_reconciliation_with_unknown_investments_is_valid_json(
    investments: pd.Series,
) -> None:
    frame = pd.DataFrame({"edicion": [2024], "inversion_mmusd": investments})
    finding = declared.verify_reference(Path("missing.pdf"), reference_row())

    result = declared.reconcile(frame, finding)

    assert result["estado"] == "revisar"
    assert result["suma_mmusd"] is None
    json.dumps(result, allow_nan=False)
