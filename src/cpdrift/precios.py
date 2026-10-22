"""First-semester copper prices aligned to portfolio editions."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PERIOD_PATTERN = re.compile(r"^(\d{4})M(\d{2})$")
PRICE_COLUMNS = ["edicion", "precio_nominal", "precio_real_2025", "meses"]


def _find_copper_column(sheet: pd.DataFrame) -> int:
    matches: list[int] = []
    for row_index in range(min(8, len(sheet) - 1)):
        for column in range(sheet.shape[1]):
            label = str(sheet.iat[row_index, column]).strip()
            unit = str(sheet.iat[row_index + 1, column]).strip()
            if label == "Copper" and unit == "($/mt)":
                matches.append(column)
    if len(matches) != 1:
        raise ValueError(f"columna Copper ($/mt) no única: {len(matches)}")
    return matches[0]


def _find_period_column(sheet: pd.DataFrame) -> int:
    matches = []
    for column in range(sheet.shape[1]):
        count = sum(
            bool(PERIOD_PATTERN.fullmatch(str(value).strip()))
            for value in sheet.iloc[:, column]
            if pd.notna(value)
        )
        if count:
            matches.append((count, column))
    if not matches:
        raise ValueError("no se encontraron períodos YYYYMmm")
    return max(matches)[1]


def build_copper_prices(
    workbook_path: Path,
    cpi_path: Path,
    editions: list[int] | set[int],
) -> pd.DataFrame:
    """Calculate January–June nominal and 2025-real prices by edition."""
    sheet = pd.read_excel(workbook_path, sheet_name="Monthly Prices", header=None)
    copper_column = _find_copper_column(sheet)
    period_column = _find_period_column(sheet)
    monthly: dict[tuple[int, int], float] = {}
    for period, price in zip(
        sheet.iloc[:, period_column], sheet.iloc[:, copper_column], strict=True
    ):
        match = PERIOD_PATTERN.fullmatch(str(period).strip())
        if match and pd.notna(price):
            monthly[(int(match.group(1)), int(match.group(2)))] = float(price)
    cpi_frame = pd.read_csv(cpi_path)
    cpi_s1 = {int(row.anio): float(row.s1) for row in cpi_frame.itertuples()}
    annual_2025_rows = cpi_frame.loc[cpi_frame["anio"].eq(2025)]
    if len(annual_2025_rows) != 1:
        raise ValueError("CPI 2025 ausente o duplicado")
    cpi_2025 = float(
        (annual_2025_rows.iloc[0]["s1"] + annual_2025_rows.iloc[0]["s2"]) / 2
    )
    rows: list[dict[str, float | int]] = []
    for edition in sorted({int(value) for value in editions}):
        if edition not in cpi_s1:
            raise ValueError(f"falta CPI s1 para edición {edition}")
        values = [
            monthly[(edition, month)]
            for month in range(1, 7)
            if (edition, month) in monthly
        ]
        if len(values) != 6:
            raise ValueError(f"edición {edition}: se encontraron {len(values)} meses")
        nominal = sum(values) / 6
        rows.append(
            {
                "edicion": edition,
                "precio_nominal": nominal,
                "precio_real_2025": nominal * cpi_2025 / cpi_s1[edition],
                "meses": 6,
            }
        )
    return pd.DataFrame(rows, columns=PRICE_COLUMNS)


def write_copper_prices(table: pd.DataFrame, interim_dir: Path) -> Path:
    """Write edition-level copper prices."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    path = interim_dir / "precio_cobre_edicion.parquet"
    table.to_parquet(path, index=False)
    return path
