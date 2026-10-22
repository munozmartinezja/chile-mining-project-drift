from pathlib import Path

import pandas as pd
import pytest

from cpdrift.precios import build_copper_prices


def price_book(path: Path, *, missing_month: int | None = None) -> None:
    rows: list[list[object]] = [
        ["metadata", None, None],
        [None, "Other", "Copper"],
        [None, "($/bbl)", "($/mt)"],
        ["period", None, None],
    ]
    for month in range(1, 7):
        if month != missing_month:
            rows.append([f"2014M{month:02d}", 10 + month, 100 + month])
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(rows).to_excel(
            writer, sheet_name="Monthly Prices", header=False, index=False
        )


def cpi_file(path: Path) -> None:
    pd.DataFrame(
        {"anio": [2014, 2025], "s1": [100.0, 200.0], "s2": [110.0, 220.0]}
    ).to_csv(path, index=False)


def test_locates_copper_header_and_deflates_first_semester(tmp_path: Path) -> None:
    workbook = tmp_path / "prices.xlsx"
    cpi = tmp_path / "cpi.csv"
    price_book(workbook)
    cpi_file(cpi)

    result = build_copper_prices(workbook, cpi, [2014])

    assert result.loc[0, "precio_nominal"] == pytest.approx(103.5)
    assert result.loc[0, "precio_real_2025"] == pytest.approx(103.5 * 210 / 100)
    assert result.loc[0, "meses"] == 6


def test_missing_copper_month_is_an_error(tmp_path: Path) -> None:
    workbook = tmp_path / "prices.xlsx"
    cpi = tmp_path / "cpi.csv"
    price_book(workbook, missing_month=4)
    cpi_file(cpi)

    with pytest.raises(ValueError, match=r"2014.*meses"):
        build_copper_prices(workbook, cpi, [2014])
