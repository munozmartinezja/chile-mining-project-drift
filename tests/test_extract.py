import pandas as pd
import pytest

from cpdrift import extract
from cpdrift.extract import join_split_rows, parse_chilean_number, parse_commissioning


def apply_corrections(frame: pd.DataFrame, path) -> pd.DataFrame:
    function = getattr(extract, "apply_extraction_corrections", None)
    assert function is not None, "falta apply_extraction_corrections"
    return function(frame, path)


def test_parse_chilean_numbers() -> None:
    assert parse_chilean_number("2.888") == 2888
    assert parse_chilean_number("1.234,5") == 1234.5


def test_parse_commissioning_keeps_range_literal() -> None:
    assert parse_commissioning("2025-2026") == (2025, "2025-2026")
    assert parse_commissioning("2025") == (2025, pd.NA)


def test_join_split_2024_row_name_across_three_lines() -> None:
    rows = [
        ["2028", "OPERADOR", "AMPLIACION", "Cobre", "BASE", "Brownfield", "Antofagasta", "1", "100"],
        [None, None, "DEPOSITO DE", None, None, None, None, None, None],
        [None, None, "RELAVES", None, None, None, None, None, None],
    ]
    assert join_split_rows(rows, 9)[0][2] == "AMPLIACION DEPOSITO DE RELAVES"


def test_extraction_correction_changes_exact_value_and_marks_row(tmp_path) -> None:
    corrections_path = tmp_path / "corrections.csv"
    corrections_path.write_text(
        "edicion,campo,valor_extraido,valor_corregido,pagina,nota\n"
        "2015,proyecto_raw,NOMBRE DAÑADO,Nombre corregido,17,lectura superpuesta\n",
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        [
            {"edicion": 2015, "proyecto_raw": "NOMBRE DAÑADO", "operador_raw": "UNO"},
            {"edicion": 2016, "proyecto_raw": "NOMBRE DAÑADO", "operador_raw": "DOS"},
        ]
    )

    corrected = apply_corrections(frame, corrections_path)

    assert corrected["proyecto_raw"].tolist() == ["Nombre corregido", "NOMBRE DAÑADO"]
    assert corrected["corregido"].tolist() == [True, False]


def test_obsolete_extraction_correction_raises(tmp_path) -> None:
    corrections_path = tmp_path / "corrections.csv"
    corrections_path.write_text(
        "edicion,campo,valor_extraido,valor_corregido,pagina,nota\n"
        "2015,operador_raw,VALOR AUSENTE,Valor corregido,17,lectura superpuesta\n",
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        [{"edicion": 2015, "proyecto_raw": "ALFA", "operador_raw": "OPERADOR"}]
    )

    with pytest.raises(
        ValueError,
        match=r"corrección obsoleta.*2015.*operador_raw.*VALOR AUSENTE",
    ):
        apply_corrections(frame, corrections_path)
