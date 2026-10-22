"""Drift indicators derived from longitudinal project appearances.

The deflator uses CPI-U, U.S. city average, all items, NSA, 1982-84=100,
semiannual series CUUS0000SA0 (FRED). The annual value is ``(s1 + s2) / 2``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TRANSITION_COLUMNS = [
    "project_id",
    "ed_prev",
    "ed_next",
    "gap_anios",
    "pem_prev",
    "pem_next",
    "delta_pem",
    "tasa_deriva",
    "capex_prev_nom",
    "capex_next_nom",
    "delta_capex_nom_pct",
    "capex_prev_real",
    "capex_next_real",
    "delta_capex_real_pct",
    "condicion_prev",
    "condicion_next",
    "sector",
    "region",
]

PROJECT_COLUMNS = [
    "project_id",
    "primera_edicion",
    "ultima_edicion",
    "n_apariciones",
    "pem_inicial",
    "pem_final",
    "deriva_acumulada",
    "anios_observados",
    "tasa_acumulada",
    "capex_inicial_nom",
    "capex_inicial_real",
    "capex_final_nom",
    "capex_final_real",
    "delta_capex_real_pct",
    "condicion_inicial",
    "sector",
    "region",
    "desenlace",
    "reingreso",
]

ENTRY_EXIT_COLUMNS = [
    "ed_prev",
    "ed_next",
    "n_base",
    "salidas_operacion",
    "salidas_sin_operar",
    "entradas_nuevas",
    "reingresos",
]
AGGREGATE_PREFIX = "OTROS"
AGGREGATE_CONTAINS = "PROYECTOS DE INFORMACION"
AGGREGATE_EXACT_NAMES = frozenset({"PROYECTOS FURE", "TRANQUES"})


@dataclass(frozen=True)
class IndicatorTables:
    """All indicator outputs and the aggregate-row audit count."""

    transiciones: pd.DataFrame
    proyectos_deriva: pd.DataFrame
    entradas_salidas: pd.DataFrame
    resumenes: dict[str, pd.DataFrame]
    fiabilidad: dict[str, float | int | None]
    n_filas_agregadas_excluidas: int


def load_annual_cpi(path: Path) -> dict[int, float]:
    """Load annual CPI averages from the fixed semiannual reference."""
    frame = pd.read_csv(path)
    required = {"anio", "s1", "s2"}
    if not required.issubset(frame.columns):
        missing = ", ".join(sorted(required - set(frame.columns)))
        raise ValueError(f"columnas CPI ausentes: {missing}")
    return {
        int(row.anio): (float(row.s1) + float(row.s2)) / 2
        for row in frame.itertuples(index=False)
    }


def _is_aggregate(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    name = str(value)
    return (
        name.startswith(AGGREGATE_PREFIX)
        or AGGREGATE_CONTAINS in name
        or name in AGGREGATE_EXACT_NAMES
    )


def _with_real_capex(
    panel: pd.DataFrame, cpi: dict[int, float]
) -> tuple[pd.DataFrame, int]:
    marked = panel.copy()
    marked["es_agregado"] = marked["nombre_norm"].map(_is_aggregate)
    result = marked.loc[~marked["es_agregado"]].copy()
    editions = {int(value) for value in result["edicion"].dropna().unique()}
    missing = sorted((editions | {2025}) - set(cpi))
    if missing:
        years = ", ".join(str(value) for value in missing)
        raise ValueError(f"faltan valores CPI para: {years}")
    annual_2025 = cpi[2025]
    ratios = result["edicion"].map(lambda value: annual_2025 / cpi[int(value)])
    result["capex_real_2025"] = pd.to_numeric(
        result["inversion_mmusd"], errors="coerce"
    ) * ratios
    return result, int(marked["es_agregado"].sum())


def _difference(left: object, right: object) -> float | None:
    if pd.isna(left) or pd.isna(right):
        return None
    return float(right) - float(left)


def _pct_change(left: object, right: object) -> float | None:
    if pd.isna(left) or pd.isna(right) or float(left) == 0:
        return None
    return (float(right) / float(left) - 1) * 100


def _build_transitions(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for project_id, appearances in panel.groupby("project_id", sort=False):
        ordered = appearances.sort_values("edicion", kind="stable")
        for (_, previous), (_, following) in itertools.pairwise(ordered.iterrows()):
            ed_prev = int(previous["edicion"])
            ed_next = int(following["edicion"])
            gap = ed_next - ed_prev
            delta_pem = _difference(
                previous["puesta_en_marcha"], following["puesta_en_marcha"]
            )
            rows.append(
                {
                    "project_id": project_id,
                    "ed_prev": ed_prev,
                    "ed_next": ed_next,
                    "gap_anios": gap,
                    "pem_prev": previous["puesta_en_marcha"],
                    "pem_next": following["puesta_en_marcha"],
                    "delta_pem": delta_pem,
                    "tasa_deriva": None if delta_pem is None else delta_pem / gap,
                    "capex_prev_nom": previous["inversion_mmusd"],
                    "capex_next_nom": following["inversion_mmusd"],
                    "delta_capex_nom_pct": _pct_change(
                        previous["inversion_mmusd"], following["inversion_mmusd"]
                    ),
                    "capex_prev_real": previous["capex_real_2025"],
                    "capex_next_real": following["capex_real_2025"],
                    "delta_capex_real_pct": _pct_change(
                        previous["capex_real_2025"], following["capex_real_2025"]
                    ),
                    "condicion_prev": previous["condicion"],
                    "condicion_next": following["condicion"],
                    "sector": previous["sector"],
                    "region": previous["region"],
                }
            )
    return pd.DataFrame(rows, columns=TRANSITION_COLUMNS)


def _build_project_drift(
    panel: pd.DataFrame, projects: pd.DataFrame
) -> pd.DataFrame:
    reference = projects.drop_duplicates("project_id").set_index("project_id")
    rows: list[dict[str, object]] = []
    for project_id, appearances in panel.groupby("project_id", sort=False):
        ordered = appearances.sort_values("edicion", kind="stable")
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        first_edition = int(first["edicion"])
        last_edition = int(last["edicion"])
        observed_years = last_edition - first_edition
        drift = _difference(first["puesta_en_marcha"], last["puesta_en_marcha"])
        if project_id not in reference.index:
            raise ValueError(f"project_id sin fila en proyectos.parquet: {project_id}")
        project = reference.loc[project_id]
        rows.append(
            {
                "project_id": project_id,
                "primera_edicion": first_edition,
                "ultima_edicion": last_edition,
                "n_apariciones": len(ordered),
                "pem_inicial": first["puesta_en_marcha"],
                "pem_final": last["puesta_en_marcha"],
                "deriva_acumulada": drift,
                "anios_observados": observed_years,
                "tasa_acumulada": (
                    None if observed_years == 0 or drift is None else drift / observed_years
                ),
                "capex_inicial_nom": first["inversion_mmusd"],
                "capex_inicial_real": first["capex_real_2025"],
                "capex_final_nom": last["inversion_mmusd"],
                "capex_final_real": last["capex_real_2025"],
                "delta_capex_real_pct": _pct_change(
                    first["capex_real_2025"], last["capex_real_2025"]
                ),
                "condicion_inicial": first["condicion"],
                "sector": first["sector"],
                "region": first["region"],
                "desenlace": project["desenlace"],
                "reingreso": bool(project["reingreso"]),
            }
        )
    return pd.DataFrame(rows, columns=PROJECT_COLUMNS)


def _build_reliability(projects: pd.DataFrame) -> dict[str, float | int | None]:
    eligible = projects.loc[
        projects["desenlace"].eq("entro_en_operacion")
        & projects["n_apariciones"].ge(2)
        & projects["deriva_acumulada"].notna()
    ]
    count = len(eligible)
    if count == 0:
        return {
            "n_proyectos": 0,
            "pct_a_tiempo": None,
            "pct_hasta_1_anio": None,
            "mediana_deriva_acumulada": None,
        }
    drift = eligible["deriva_acumulada"].astype(float)
    return {
        "n_proyectos": count,
        "pct_a_tiempo": float(drift.le(0).mean() * 100),
        "pct_hasta_1_anio": float(drift.le(1).mean() * 100),
        "mediana_deriva_acumulada": float(drift.median()),
    }


def _build_entries_exits(
    panel: pd.DataFrame,
    egresos: pd.DataFrame,
) -> pd.DataFrame:
    eligible = panel.loc[pd.to_numeric(panel["inversion_mmusd"], errors="coerce").ge(90)]
    editions = sorted(int(value) for value in panel["edicion"].unique())
    seen: set[object] = set()
    rows: list[dict[str, int]] = []
    by_edition = {
        edition: group.drop_duplicates("project_id").set_index("project_id")
        for edition, group in eligible.groupby("edicion", sort=False)
    }
    all_ids = {
        int(edition): set(group["project_id"])
        for edition, group in panel.groupby("edicion", sort=False)
    }
    for ed_prev, ed_next in itertools.pairwise(editions):
        previous = by_edition.get(ed_prev, pd.DataFrame())
        following = by_edition.get(ed_next, pd.DataFrame())
        previous_ids = set(previous.index)
        following_ids = set(following.index)
        exiting = previous_ids - all_ids.get(ed_next, set())
        entering = following_ids - previous_ids
        operation = 0
        without_operation = 0
        for project_id in exiting:
            raw_name = previous.loc[project_id, "proyecto_raw"]
            matches = egresos.loc[
                egresos["edicion_desde"].eq(ed_prev)
                & egresos["edicion_hasta"].eq(ed_next)
                & egresos["proyecto_raw"].eq(raw_name)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"egreso no único para {ed_prev} / {raw_name}: {len(matches)}"
                )
            reason = str(matches.iloc[0]["motivo"])
            operation += int(reason == "operacion")
            without_operation += int(reason == "reestructuracion")
        rows.append(
            {
                "ed_prev": ed_prev,
                "ed_next": ed_next,
                "n_base": len(previous_ids),
                "salidas_operacion": operation,
                "salidas_sin_operar": without_operation,
                "entradas_nuevas": len(entering - seen),
                "reingresos": len(entering & seen),
            }
        )
        seen.update(previous_ids)
    return pd.DataFrame(rows, columns=ENTRY_EXIT_COLUMNS)


def _build_summaries(
    transitions: pd.DataFrame, projects: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    source = transitions.copy()
    initial_conditions = projects.set_index("project_id")["condicion_inicial"]
    source["condicion_inicial"] = source["project_id"].map(initial_conditions)
    summaries: dict[str, pd.DataFrame] = {}
    for dimension in ("condicion_inicial", "sector", "region"):
        rows: list[dict[str, object]] = []
        for value, group in source.groupby(dimension, dropna=False, sort=True):
            delta = pd.to_numeric(group["delta_pem"], errors="coerce").dropna()
            rates = pd.to_numeric(group["tasa_deriva"], errors="coerce").dropna()
            capex = pd.to_numeric(
                group["delta_capex_real_pct"], errors="coerce"
            ).dropna()
            rows.append(
                {
                    dimension: value,
                    "n": len(group),
                    "mediana_tasa_deriva": (
                        None if rates.empty else float(rates.median())
                    ),
                    "media_tasa_deriva": None if rates.empty else float(rates.mean()),
                    "pct_transiciones_atraso": (
                        None if delta.empty else float(delta.gt(0).mean() * 100)
                    ),
                    "mediana_delta_capex_real_pct": (
                        None if capex.empty else float(capex.median())
                    ),
                }
            )
        summaries[dimension] = pd.DataFrame(
            rows,
            columns=[
                dimension,
                "n",
                "mediana_tasa_deriva",
                "media_tasa_deriva",
                "pct_transiciones_atraso",
                "mediana_delta_capex_real_pct",
            ],
        )
    return summaries


def build_indicator_tables(
    panel: pd.DataFrame,
    projects: pd.DataFrame,
    cpi_path: Path,
    egresos: pd.DataFrame | None = None,
) -> IndicatorTables:
    """Build all drift tables without mutating the input frames."""
    filtered, excluded = _with_real_capex(panel, load_annual_cpi(cpi_path))
    transitions = _build_transitions(filtered)
    project_drift = _build_project_drift(filtered, projects)
    return IndicatorTables(
        transiciones=transitions,
        proyectos_deriva=project_drift,
        entradas_salidas=_build_entries_exits(
            filtered,
            pd.DataFrame() if egresos is None else egresos,
        ),
        resumenes=_build_summaries(transitions, project_drift),
        fiabilidad=_build_reliability(project_drift),
        n_filas_agregadas_excluidas=excluded,
    )


def write_indicator_tables(tables: IndicatorTables, interim_dir: Path) -> None:
    """Write the indicator tables to the interim output directory."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    tables.transiciones.to_parquet(interim_dir / "transiciones.parquet", index=False)
    tables.proyectos_deriva.to_parquet(
        interim_dir / "proyectos_deriva.parquet", index=False
    )
    tables.entradas_salidas.to_parquet(
        interim_dir / "entradas_salidas.parquet", index=False
    )
    for dimension, frame in tables.resumenes.items():
        frame.to_parquet(interim_dir / f"resumen_{dimension}.parquet", index=False)
