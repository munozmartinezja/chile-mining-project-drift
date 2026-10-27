"""Extract project-level rows from the supplied Cochilco source files."""

from __future__ import annotations

import math
import re
from itertools import pairwise
from pathlib import Path
from typing import Any

import pandas as pd

from .declared import citation_in_pages
from .ediciones import DATA_EDITIONS
from .pdftext import get_pages_text, get_pdf_data

CANONICAL_COLUMNS = [
    "edicion",
    "puesta_en_marcha",
    "puesta_en_marcha_raw",
    "proyecto_raw",
    "operador_raw",
    "sector",
    "region",
    "tipo_proyecto",
    "condicion",
    "etapa",
    "estado_permisos",
    "inversion_mmusd",
    "rotulo_inversion",
    "pagina_fuente",
]

TRANSCRIPTION_2022_COLUMNS = [
    "edicion", "puesta_en_marcha_raw", "proyecto", "operador", "sector",
    "region", "tipo_proyecto", "condicion", "etapa", "estado_permisos",
    "inversion_mmusd", "pagina",
]
CORRECTION_COLUMNS = [
    "edicion",
    "campo",
    "valor_extraido",
    "valor_corregido",
    "pagina",
    "nota",
]
PDF_TABLE_PAGES = {
    2014: [52],
    2015: [17],
    2016: [13],
    2017: [13],
    2018: [14],
    2019: [14],
    2020: [14],
    2021: [13],
    2023: [31, 32, 33],
    2024: [10, 11, 12, 13, 14],
}
WORD_COLUMN_EDGES = {
    2014: [0, 65, 170, 275, 326, 370, 415, 452, 496, 540, 612],
    2015: [0, 54, 168, 260, 324, 372, 416, 459, 507, 557, 612],
    2016: [0, 51, 185, 291, 342, 380, 423, 464, 505, 566, 612],
    2017: [0, 58, 171, 263, 313, 356, 399, 441, 483, 573, 612],
    2018: [0, 55, 183, 285, 337, 379, 419, 459, 502, 576, 612],
    2019: [0, 56, 189, 300, 348, 391, 431, 475, 517, 570, 612],
    2020: [0, 60, 190, 292, 341, 384, 430, 475, 518, 568, 612],
    2021: [0, 73, 209, 314, 359, 396, 432, 471, 514, 560, 612],
}


def _clean(value: object) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def parse_chilean_number(value: object) -> float:
    """Parse dot-thousands and comma-decimal notation used in Chile."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if pd.isna(value):
            raise ValueError("number is null")
        return float(value)
    text = _clean(value).replace("MMUS$", "").replace("US$", "").replace("$", "")
    text = re.sub(r"[^0-9,.-]", "", text)
    if not text:
        raise ValueError(f"not a Chilean number: {value!r}")
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+", text):
        text = text.replace(".", "")
    return float(text)


def parse_commissioning(value: object) -> tuple[int | None, object]:
    literal = _clean(value)
    years = re.findall(r"\b20\d{2}\b", literal)
    if not years:
        return None, pd.NA
    raw: object = literal if len(years) > 1 else pd.NA
    return int(years[0]), raw


def join_split_rows(rows: list[list[object]], expected_columns: int) -> list[list[object]]:
    """Join physical continuation lines onto the preceding logical record."""
    result: list[list[object]] = []
    for raw in rows:
        row = (list(raw) + [None] * expected_columns)[:expected_columns]
        first = _clean(row[0])
        if first.casefold().startswith(("total", "fuente")):
            continue
        starts = bool(re.search(r"\b20\d{2}\b", first))
        if starts:
            result.append([_clean(value) or None for value in row])
        elif result and any(_clean(value) for value in row):
            for index, value in enumerate(row):
                continuation = _clean(value)
                if continuation:
                    previous = _clean(result[-1][index])
                    result[-1][index] = f"{previous} {continuation}".strip()
    return result


def _table_records(tables: list[list[list[object]]]) -> list[list[str]]:
    records: list[list[str]] = []
    groups = [
        (0, 3), (3, 5), (5, 7), (7, 9), (9, 11),
        (11, 13), (13, 15), (15, 17), (17, 18), (18, 20),
    ]
    for table in tables:
        if max((len(row) for row in table), default=0) not in {19, 20}:
            continue
        for raw in table:
            padded = list(raw) + [None] * (20 - len(raw))
            row = [" ".join(_clean(value) for value in padded[a:b] if _clean(value)) for a, b in groups]
            row[0] = re.sub(r"(?<=\d)\s+(?=\d)", "", row[0])
            if re.search(r"\b20\d{2}\b", row[0]):
                records.append(row)
            elif records and any(row):
                for index, value in enumerate(row):
                    if value:
                        records[-1][index] = f"{records[-1][index]} {value}".strip()
    return records


def _word_records(
    words: list[dict[str, Any]], edition: int, page_height: float
) -> list[list[str]]:
    edges = WORD_COLUMN_EDGES[edition]
    amount_words = [
        word for word in words
        if edges[-2] <= word["x0"] < edges[-1]
        and re.fullmatch(r"[\d.]+", str(word["text"]))
    ]
    candidates: dict[float, tuple[float, float]] = {}
    for word in words:
        if not edges[0] <= word["x0"] < edges[1]:
            continue
        if re.fullmatch(r"20\d{2}", str(word["text"])) is None:
            continue
        nearby = sorted(amount_words, key=lambda item: abs(float(item["top"]) - float(word["top"])))
        if not nearby or abs(float(nearby[0]["top"]) - float(word["top"])) > 6:
            continue
        amount_top = float(nearby[0]["top"])
        distance = abs(amount_top - float(word["top"]))
        if amount_top not in candidates or distance < candidates[amount_top][1]:
            candidates[amount_top] = (float(word["top"]), distance)
    tops = sorted(value[0] for value in candidates.values())
    records: list[list[str]] = []
    for position, top in enumerate(tops):
        upper = (tops[position - 1] + top) / 2 if position else top - 7
        if position + 1 < len(tops):
            lower = (top + tops[position + 1]) / 2 + 0.5
        else:
            endings = [
                float(word["top"])
                for word in words
                if float(word["top"]) > top
                and str(word["text"]).casefold().startswith(("total", "fuente"))
            ]
            lower = min(endings) if endings else page_height - 45
        row = []
        for left, right in pairwise(edges):
            cell = [
                word for word in words
                if left <= word["x0"] < right and upper <= word["top"] < lower
            ]
            cell.sort(key=lambda item: (round(float(item["top"]), 1), item["x0"]))
            row.append(" ".join(str(item["text"]) for item in cell))
        if len(row) == 10:
            records.append(row)
    return records


def _record(
    edition: int,
    row: list[object],
    label: str,
    page: int | None,
    layout: str = "legacy",
) -> dict[str, object]:
    if layout == "2023":
        year, operator, project, sector, condition, amount = row
        region = project_type = stage = permits = None
    elif layout == "2024":
        year, operator, project, sector, condition, project_type, region, _, amount = row
        stage = permits = None
    else:
        year, project, operator, sector, region, project_type, condition, stage, permits, amount = row
    commissioning, commissioning_raw = parse_commissioning(year)
    try:
        investment = parse_chilean_number(amount)
    except ValueError:
        investment = math.nan
    return {
        "edicion": edition,
        "puesta_en_marcha": commissioning,
        "puesta_en_marcha_raw": commissioning_raw,
        "proyecto_raw": _clean(project),
        "operador_raw": _clean(operator),
        "sector": _clean(sector) or pd.NA,
        "region": _clean(region) or pd.NA,
        "tipo_proyecto": _clean(project_type) or pd.NA,
        "condicion": _clean(condition) or pd.NA,
        "etapa": _clean(stage) or pd.NA,
        "estado_permisos": _clean(permits) or pd.NA,
        "inversion_mmusd": investment,
        "rotulo_inversion": label,
        "pagina_fuente": page,
    }


def extract_pdf_edition(path: Path, edition: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, object]] = []
    methods: set[str] = set()
    pages = get_pdf_data(path)["pages"]
    for page_number in PDF_TABLE_PAGES[edition]:
        page = pages[page_number - 1]
        tables = page["tables"]
        if edition == 2024:
            rows = join_split_rows(tables[0][1:] if tables else [], 9)
            label = "Inversión 2024-2033 (MMUS$)"
            records.extend(
                _record(edition, row, label, page_number, "2024") for row in rows
            )
            methods.add("extract_tables")
        elif edition == 2023:
            table = max(tables, key=len) if tables else []
            rows = join_split_rows([row[:6] for row in table], 6)
            label = "Inversión 2023-2032 (MMUS$)"
            records.extend(
                _record(edition, row, label, page_number, "2023") for row in rows
            )
            methods.add("extract_tables")
        else:
            table_rows = _table_records(tables)
            word_rows = _word_records(page["words"], edition, float(page["height"]))
            rows = table_rows if len(table_rows) >= len(word_rows) else word_rows
            methods.add("extract_tables" if rows is table_rows else "extract_words")
            labels = {
                2014: "Inversión 07/2014 (MMUS$)",
                2015: "Inversión a Julio de 2015 (MMUS$)",
                2016: "Inversión a Agosto de 2016 (MMUS$)",
            }
            label = labels.get(edition, "Inversión (MMUS$)")
            records.extend(_record(edition, row, label, page_number) for row in rows)
    frame = pd.DataFrame(records, columns=CANONICAL_COLUMNS)
    frame = frame.loc[frame["proyecto_raw"].ne("")].reset_index(drop=True)
    return frame, {
        "pages": PDF_TABLE_PAGES[edition],
        "method": "+".join(sorted(methods)),
    }


def _key(value: object) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKD", _clean(value).casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", text))


def extract_2025_workbook(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object, engine="openpyxl")
    candidates: list[tuple[int, str, pd.DataFrame]] = []
    for sheet_name, sheet in sheets.items():
        for index, source in sheet.iterrows():
            values = [_clean(value) for value in source.tolist()]
            if "Puesta en Marcha" not in values or "Nombre del Proyecto" not in values:
                continue
            selected = [column for column, value in enumerate(values) if value]
            headers = [values[column] for column in selected]
            data: list[list[object]] = []
            for _, raw in sheet.iloc[index + 1 :].iterrows():
                row = [raw.iloc[column] for column in selected]
                if not any(_clean(value) for value in row):
                    continue
                if _clean(row[0]).startswith(("Fuente:", "(")):
                    break
                data.append(row)
            candidates.append((len(data), sheet_name, pd.DataFrame(data, columns=headers)))
            break
    if not candidates:
        raise ValueError("2025 workbook project header not found")
    _, sheet_name, raw = max(candidates, key=lambda item: item[0])
    columns = {_key(column): column for column in raw.columns}

    def column(*names: str) -> str | None:
        return next((columns[name] for name in names if name in columns), None)

    amount = column("inversion mmus")
    if amount is None:
        raise ValueError("2025 workbook investment column not found")
    records = []
    for source in raw.to_dict("records"):
        row = [
            source.get(column("puesta en marcha")),
            source.get(column("nombre del proyecto")),
            source.get(column("empresa", "operador")),
            source.get(column("sector")),
            source.get(column("region")),
            source.get(column("tipo de proyecto")),
            source.get(column("condicion")),
            source.get(column("etapa de avance", "etapa de desarrollo")),
            source.get(column("estado de permisos ambientales")),
            source.get(amount),
        ]
        records.append(_record(2025, row, str(amount), None))
    return pd.DataFrame(records, columns=CANONICAL_COLUMNS), {
        "source": "xlsx",
        "sheet": sheet_name,
    }


def extract_2022_transcription(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the curated 2022 transcription through the legacy row parser."""
    source = pd.read_csv(path, dtype=str, keep_default_na=False)
    if source.columns.tolist() != TRANSCRIPTION_2022_COLUMNS:
        raise ValueError(f"columnas inválidas en transcripción 2022: {path}")
    if not source["edicion"].eq("2022").all():
        raise ValueError("la transcripción 2022 contiene otra edición")
    records = []
    for row in source.itertuples(index=False):
        legacy = [
            row.puesta_en_marcha_raw, row.proyecto, row.operador, row.sector,
            row.region, row.tipo_proyecto, row.condicion, row.etapa,
            row.estado_permisos, row.inversion_mmusd,
        ]
        records.append(_record(2022, legacy, "Inversión (MMUS$)", int(row.pagina)))
    return pd.DataFrame(records, columns=CANONICAL_COLUMNS), {
        "pages": [10],
        "method": "transcripcion",
    }


def verify_transcription_2022(
    extracted: pd.DataFrame, subtotales_path: Path, raw_dir: Path
) -> None:
    """Reject an incomplete or source-inconsistent 2022 transcription."""
    rows = extracted.loc[extracted["edicion"].eq(2022)]
    if len(rows) != 53:
        raise ValueError(f"transcripción 2022 requiere 53 filas, recibió {len(rows)}")
    subtotals = pd.read_csv(subtotales_path, dtype=str, keep_default_na=False)
    expected = ["dimension", "valor", "n_declarado", "capex_mmusd", "pagina", "frase"]
    if subtotals.columns.tolist() != expected:
        raise ValueError(f"columnas inválidas en subtotales 2022: {subtotales_path}")
    pages = get_pages_text(raw_dir / "cartera_2022.pdf")
    for subtotal in subtotals.itertuples(index=False):
        if subtotal.dimension not in {"condicion", "sector"}:
            raise ValueError(f"dimensión inválida en subtotal 2022: {subtotal.dimension}")
        selected = rows.loc[rows[subtotal.dimension].eq(subtotal.valor)]
        observed_count = len(selected)
        observed_capex = float(selected["inversion_mmusd"].sum())
        expected_count = int(subtotal.n_declarado)
        expected_capex = parse_chilean_number(subtotal.capex_mmusd)
        if observed_count != expected_count or observed_capex != expected_capex:
            raise ValueError(
                f"subtotal 2022 no cuadra para {subtotal.dimension}={subtotal.valor!r}: "
                f"({observed_count}, {observed_capex}) != "
                f"({expected_count}, {expected_capex})"
            )
        found, searched = citation_in_pages(pages, subtotal.frase, int(subtotal.pagina))
        if not found:
            raise ValueError(
                f"cita de subtotal 2022 no encontrada en páginas {searched}: "
                f"{subtotal.dimension}={subtotal.valor!r}"
            )


def extract_all(raw_dir: Path, reference_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    provenance: dict[str, Any] = {}
    for edition in DATA_EDITIONS[:-1]:
        if edition == 2022:
            frame, detail = extract_2022_transcription(
                reference_dir / "cartera_2022_transcrita.csv"
            )
        else:
            frame, detail = extract_pdf_edition(raw_dir / f"cartera_{edition}.pdf", edition)
        frames.append(frame)
        provenance[str(edition)] = detail
    frame, detail = extract_2025_workbook(raw_dir / "anexo_c_2025.xlsx")
    frames.append(frame)
    provenance["2025"] = detail
    result = pd.concat(frames, ignore_index=True)
    result["edicion"] = result["edicion"].astype("int64")
    result["puesta_en_marcha"] = pd.array(result["puesta_en_marcha"], dtype="Int64")
    result["inversion_mmusd"] = pd.to_numeric(result["inversion_mmusd"], errors="coerce")
    verify_transcription_2022(result, reference_dir / "subtotales_2022.csv", raw_dir)
    return result, provenance


def apply_extraction_corrections(
    frame: pd.DataFrame, corrections_path: Path
) -> pd.DataFrame:
    """Apply exact, edition-scoped source corrections and reject stale entries."""
    corrections = pd.read_csv(corrections_path, dtype=str, keep_default_na=False)
    if corrections.columns.tolist() != CORRECTION_COLUMNS:
        raise ValueError(f"columnas inválidas en correcciones de extracción: {corrections_path}")
    result = frame.copy()
    result["corregido"] = False
    for correction in corrections.to_dict("records"):
        edition = int(correction["edicion"])
        field = correction["campo"]
        extracted = correction["valor_extraido"]
        if field not in result.columns:
            raise ValueError(f"campo desconocido en corrección de extracción: {field}")
        matches = result["edicion"].eq(edition) & result[field].eq(extracted)
        if not matches.any():
            raise ValueError(
                "corrección obsoleta: "
                f"edición {edition}, campo {field}, valor {extracted!r}"
            )
        result.loc[matches, field] = correction["valor_corregido"]
        result.loc[matches, "corregido"] = True
    return result
