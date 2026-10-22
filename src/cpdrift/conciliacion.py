"""Verified reconciliation of declared portfolio changes and observed outcomes."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .declared import citation_in_pages, normalize_citation
from .pdftext import get_pages_text

CHANGE_COLUMNS = [
    "edicion",
    "par_desde",
    "categoria",
    "n_declarado",
    "capex_mmusd",
    "pagina",
    "frase",
]
EXIT_COLUMNS = [
    "edicion_desde",
    "edicion_hasta",
    "proyecto_raw",
    "motivo",
    "fuente",
    "edicion_fuente",
    "pagina",
    "frase",
    "nota",
]
ADJUSTMENT_COLUMNS = [
    "edicion_desde",
    "edicion_hasta",
    "categoria",
    "delta_n",
    "delta_capex",
    "nota",
]
OUTPUT_COLUMNS = [
    "edicion_desde",
    "edicion_hasta",
    "categoria",
    "decl_n",
    "decl_capex",
    "obs_n",
    "obs_capex",
    "delta_n",
    "delta_capex",
    "compara_capex",
    "estado",
]
CHANGE_CATEGORIES = {
    "entraron_operacion",
    "reestructuracion",
    "nuevos",
    "metodologia",
}
EXIT_REASONS = {
    "operacion",
    "reestructuracion",
    "metodologia",
    "fusion",
    "no_declarado",
}
EXIT_SOURCES = {"nombre", "cuadro", "revision"}
AGGREGATE_PREFIX = "OTROS"
AGGREGATE_CONTAINS = "PROYECTOS DE INFORMACION"
AGGREGATE_EXACT_NAMES = frozenset({"PROYECTOS FURE", "TRANQUES"})


@dataclass(frozen=True)
class ReconciliationReferences:
    """Validated reference tables used by the outcome gate."""

    cambios: pd.DataFrame
    egresos: pd.DataFrame
    ajustes: pd.DataFrame


@dataclass(frozen=True)
class DateRuleCheck:
    """Confusion matrix and row detail for the former date rule."""

    matriz: pd.DataFrame
    detalle: pd.DataFrame


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.columns.tolist() != columns:
        raise ValueError(f"esquema inválido en {path.name}: {frame.columns.tolist()}")
    return frame


def _numeric(frame: pd.DataFrame, columns: list[str], filename: str) -> None:
    for column in columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & converted.isna()
        if invalid.any():
            values = sorted(frame.loc[invalid, column].astype(str).unique())
            raise ValueError(f"{filename}: {column} no numérico: {values}")
        frame[column] = converted


def _domain(
    frame: pd.DataFrame,
    column: str,
    allowed: set[str],
    filename: str,
) -> None:
    invalid = sorted(set(frame[column].dropna().astype(str)) - allowed)
    if invalid:
        raise ValueError(f"{filename}: {column} fuera de dominio: {invalid}")


def load_reconciliation_references(
    reference_dir: Path = Path("data/reference"),
) -> ReconciliationReferences:
    """Load and validate all three curated outcome reference tables."""
    changes_name = "cambios_cartera_declarados.csv"
    exits_name = "egresos_declarados.csv"
    adjustments_name = "ajustes_conciliacion.csv"
    changes = _read_csv(reference_dir / changes_name, CHANGE_COLUMNS)
    exits = _read_csv(reference_dir / exits_name, EXIT_COLUMNS)
    adjustments = _read_csv(reference_dir / adjustments_name, ADJUSTMENT_COLUMNS)
    _numeric(
        changes,
        ["edicion", "par_desde", "n_declarado", "capex_mmusd"],
        changes_name,
    )
    _numeric(
        exits,
        ["edicion_desde", "edicion_hasta", "edicion_fuente"],
        exits_name,
    )
    _numeric(
        adjustments,
        ["edicion_desde", "edicion_hasta", "delta_n", "delta_capex"],
        adjustments_name,
    )
    _domain(changes, "categoria", CHANGE_CATEGORIES, changes_name)
    _domain(adjustments, "categoria", CHANGE_CATEGORIES, adjustments_name)
    _domain(exits, "motivo", EXIT_REASONS, exits_name)
    _domain(exits, "fuente", EXIT_SOURCES, exits_name)
    if changes[["edicion", "par_desde", "n_declarado", "capex_mmusd"]].isna().any().any():
        raise ValueError(f"{changes_name}: valores obligatorios vacíos")
    if exits[
        [
            "edicion_desde",
            "edicion_hasta",
            "proyecto_raw",
            "motivo",
            "fuente",
            "edicion_fuente",
        ]
    ].isna().any().any():
        raise ValueError(f"{exits_name}: valores obligatorios vacíos")
    if adjustments[
        ["edicion_desde", "edicion_hasta", "categoria", "delta_n"]
    ].isna().any().any():
        raise ValueError(f"{adjustments_name}: valores obligatorios vacíos")
    phrases = exits["frase"].fillna("").astype(str).str.strip()
    invalid_named = exits["fuente"].eq("nombre") & phrases.eq("")
    invalid_unnamed = exits["fuente"].isin({"cuadro", "revision"}) & phrases.ne("")
    if invalid_named.any() or invalid_unnamed.any():
        raise ValueError(f"{exits_name}: frase incompatible con fuente")
    if exits.duplicated(["edicion_desde", "proyecto_raw"]).any():
        raise ValueError(f"{exits_name}: egreso duplicado")
    if adjustments.duplicated(
        ["edicion_desde", "edicion_hasta", "categoria"]
    ).any():
        raise ValueError(f"{adjustments_name}: ajuste duplicado")
    return ReconciliationReferences(changes, exits, adjustments)


def _is_aggregate(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    name = str(value)
    return (
        name.startswith(AGGREGATE_PREFIX)
        or AGGREGATE_CONTAINS in name
        or name in AGGREGATE_EXACT_NAMES
    )


def _verify_workbook_phrase(path: Path, sheet: str, phrase: str) -> bool:
    if not path.exists():
        return False
    cells = pd.read_excel(path, sheet_name=sheet, header=None)
    expected = normalize_citation(phrase)
    return any(
        expected in normalize_citation(str(value))
        for value in cells.to_numpy().ravel()
        if pd.notna(value)
    )


def verify_citations(
    references: ReconciliationReferences,
    raw_dir: Path,
) -> None:
    """Require every non-empty curated phrase to be present at its citation."""
    citations: list[tuple[int, object, str, str]] = []
    for row in references.cambios.itertuples(index=False):
        phrase = "" if pd.isna(row.frase) else str(row.frase).strip()
        if phrase:
            citations.append((int(row.edicion), row.pagina, phrase, "cambios"))
    for row in references.egresos.itertuples(index=False):
        phrase = "" if pd.isna(row.frase) else str(row.frase).strip()
        if phrase:
            citations.append(
                (int(row.edicion_fuente), row.pagina, phrase, "egresos")
            )
    page_cache: dict[int, list[str]] = {}
    failures: list[str] = []
    for edition, page, phrase, table in citations:
        if edition == 2025 and str(page).strip() == "Tabla 2 -3":
            found = _verify_workbook_phrase(
                raw_dir / "anexo_c_2025.xlsx", "Tabla 2 -3", phrase
            )
        else:
            pdf_path = raw_dir / f"cartera_{edition}.pdf"
            if not pdf_path.exists():
                found = False
            else:
                if edition not in page_cache:
                    page_cache[edition] = get_pages_text(pdf_path)
                pages = page_cache[edition]
                try:
                    page_number = int(float(page))
                except (TypeError, ValueError):
                    raise ValueError(
                        f"cita {table} {edition} con página inválida: {page}"
                    ) from None
                found, _ = citation_in_pages(pages, phrase, page_number)
        if not found:
            failures.append(f"{table} {edition} página {page}: {phrase}")
    if failures:
        raise ValueError("citas no encontradas:\n" + "\n".join(failures))


def _pair_exits(
    panel: pd.DataFrame,
    edition_from: int,
    edition_to: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    previous = panel.loc[panel["edicion"].eq(edition_from)].drop_duplicates(
        "project_id"
    )
    following = panel.loc[panel["edicion"].eq(edition_to)].drop_duplicates(
        "project_id"
    )
    previous_ids = set(previous["project_id"])
    following_ids = set(following["project_id"])
    exits = previous.loc[previous["project_id"].isin(previous_ids - following_ids)]
    entries = following.loc[following["project_id"].isin(following_ids - previous_ids)]
    return exits, entries


def _sum_capex(frame: pd.DataFrame) -> float:
    return float(pd.to_numeric(frame["inversion_mmusd"], errors="coerce").sum())


def build_reconciliation(
    panel: pd.DataFrame,
    references: ReconciliationReferences,
) -> pd.DataFrame:
    """Build the declared-versus-observed table and validate exit identities."""
    eligible = panel.loc[~panel["nombre_norm"].map(_is_aggregate)].copy()
    adjustments = references.ajustes.reindex(columns=ADJUSTMENT_COLUMNS)
    editions = sorted(int(value) for value in eligible["edicion"].unique())
    rows: list[dict[str, object]] = []
    for edition_from, edition_to in itertools.pairwise(editions):
        exits, entries = _pair_exits(eligible, edition_from, edition_to)
        declared_exits = references.egresos.loc[
            references.egresos["edicion_desde"].eq(edition_from)
            & references.egresos["edicion_hasta"].eq(edition_to)
        ]
        observed_names = set(exits["proyecto_raw"].astype(str))
        declared_names = set(declared_exits["proyecto_raw"].astype(str))
        if observed_names != declared_names:
            missing = sorted(observed_names - declared_names)
            extra = sorted(declared_names - observed_names)
            raise ValueError(
                f"salidas distintas {edition_from}→{edition_to}; "
                f"sin egreso={missing}; sin salida={extra}"
            )
        exit_with_reason = exits.merge(
            declared_exits[["proyecto_raw", "motivo"]],
            on="proyecto_raw",
            how="left",
            validate="one_to_one",
        )
        observed: dict[str, tuple[int, float]] = {
            "nuevos": (len(entries), _sum_capex(entries))
        }
        for category, reason in (
            ("entraron_operacion", "operacion"),
            ("reestructuracion", "reestructuracion"),
            ("metodologia", "metodologia"),
        ):
            selected = exit_with_reason.loc[exit_with_reason["motivo"].eq(reason)]
            observed[category] = (len(selected), _sum_capex(selected))
        declared = references.cambios.loc[
            references.cambios["par_desde"].ge(edition_from)
            & references.cambios["edicion"].gt(edition_from)
            & references.cambios["edicion"].le(edition_to)
        ]
        categories = set(declared["categoria"].astype(str)) | {
            category for category, (count, _) in observed.items() if count > 0
        }
        for category in sorted(categories):
            declaration = declared.loc[declared["categoria"].eq(category)]
            declared_count = int(declaration["n_declarado"].sum())
            declared_capex = float(declaration["capex_mmusd"].sum())
            observed_count, observed_capex = observed.get(category, (0, 0.0))
            adjustment = adjustments.loc[
                adjustments["edicion_desde"].eq(edition_from)
                & adjustments["edicion_hasta"].eq(edition_to)
                & adjustments["categoria"].eq(category)
            ]
            if adjustment.empty:
                delta_count = 0
                delta_capex: float | None = 0.0
                compare_capex = True
            else:
                delta_count = int(adjustment.iloc[0]["delta_n"])
                raw_delta_capex = adjustment.iloc[0]["delta_capex"]
                delta_capex = (
                    None if pd.isna(raw_delta_capex) else float(raw_delta_capex)
                )
                compare_capex = delta_capex is not None
            count_ok = observed_count + delta_count == declared_count
            capex_ok = (
                not compare_capex
                or abs(observed_capex + float(delta_capex) - declared_capex) < 0.5
            )
            rows.append(
                {
                    "edicion_desde": edition_from,
                    "edicion_hasta": edition_to,
                    "categoria": category,
                    "decl_n": declared_count,
                    "decl_capex": declared_capex,
                    "obs_n": observed_count,
                    "obs_capex": observed_capex,
                    "delta_n": delta_count,
                    "delta_capex": delta_capex,
                    "compara_capex": compare_capex,
                    "estado": "ok" if count_ok and capex_ok else "falla",
                }
            )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def require_reconciled(table: pd.DataFrame) -> None:
    """Stop when any declared-versus-observed comparison fails."""
    failures = table.loc[table["estado"].eq("falla")]
    if failures.empty:
        return
    columns = [
        "edicion_desde",
        "edicion_hasta",
        "categoria",
        "decl_n",
        "obs_n",
        "delta_n",
        "decl_capex",
        "obs_capex",
        "delta_capex",
    ]
    raise ValueError("conciliación de desenlaces fallida:\n" + failures[columns].to_string(index=False))


def write_reconciliation(table: pd.DataFrame, interim_dir: Path) -> Path:
    """Write the verified outcome reconciliation table."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    path = interim_dir / "conciliacion_desenlaces.parquet"
    table.to_parquet(path, index=False)
    return path


def build_date_rule_check(panel: pd.DataFrame, egresos: pd.DataFrame) -> DateRuleCheck:
    """Compare declared outcomes with the former planned-start rule."""
    eligible = egresos.loc[
        egresos["motivo"].isin({"operacion", "reestructuracion"})
    ]
    rows = []
    for exit_row in eligible.itertuples(index=False):
        matches = panel.loc[
            panel["edicion"].eq(exit_row.edicion_desde)
            & panel["proyecto_raw"].eq(exit_row.proyecto_raw)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"fila de salida no única para {exit_row.edicion_desde} / "
                f"{exit_row.proyecto_raw}: {len(matches)}"
            )
        pem = matches.iloc[0]["puesta_en_marcha"]
        rule = (
            "operacion"
            if pd.notna(pem) and float(pem) <= int(exit_row.edicion_hasta)
            else "reestructuracion"
        )
        rows.append(
            {
                "edicion_desde": int(exit_row.edicion_desde),
                "edicion_hasta": int(exit_row.edicion_hasta),
                "proyecto_raw": exit_row.proyecto_raw,
                "pem": pem,
                "declarado": exit_row.motivo,
                "regla": rule,
                "discrepante": rule != exit_row.motivo,
            }
        )
    detail = pd.DataFrame(
        rows,
        columns=[
            "edicion_desde",
            "edicion_hasta",
            "proyecto_raw",
            "pem",
            "declarado",
            "regla",
            "discrepante",
        ],
    )
    if detail.empty:
        matrix = pd.DataFrame(columns=["declarado", "regla", "n"])
    else:
        matrix = (
            detail.groupby(["declarado", "regla"], as_index=False)
            .size()
            .rename(columns={"size": "n"})
        )
    return DateRuleCheck(matrix, detail)


def write_date_rule_check(check: DateRuleCheck, interim_dir: Path) -> Path:
    """Write matrix and detail in one typed long table."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    matrix = check.matriz.assign(tipo="matriz")
    detail = check.detalle.assign(tipo="detalle")
    table = pd.concat([matrix, detail], ignore_index=True, sort=False)
    path = interim_dir / "regla_vs_declarado.parquet"
    table.to_parquet(path, index=False)
    return path
