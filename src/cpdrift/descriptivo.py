"""Publication-ready descriptive tables for the longitudinal portfolio."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .indicators import load_annual_cpi

AGGREGATE_PREFIX = "OTROS"
AGGREGATE_CONTAINS = "PROYECTOS DE INFORMACION"
AGGREGATE_EXACT_NAMES = frozenset({"PROYECTOS FURE", "TRANQUES"})
DELTA_CLASSES = ("<=-2", "-1", "0", "1", "2", "3", "4", "5", ">=6")


def _is_aggregate(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    name = str(value)
    return (
        name.startswith(AGGREGATE_PREFIX)
        or AGGREGATE_CONTAINS in name
        or name in AGGREGATE_EXACT_NAMES
    )


def delta_class(value: object) -> str | None:
    """Return the published discrete class for a launch-year change."""
    if pd.isna(value):
        return None
    number = float(value)
    if number <= -2:
        return "<=-2"
    if number >= 6:
        return ">=6"
    return str(int(number))


def add_public_ids(projects: pd.DataFrame) -> pd.DataFrame:
    """Assign stable anonymous IDs from lifecycle fields and internal ID."""
    result = projects.copy()
    order = result.sort_values(
        ["primera_edicion", "pem_primera", "capex_primero", "project_id"],
        kind="stable",
        na_position="last",
    ).index
    identifiers = {
        index: f"C3-{position:03d}" for position, index in enumerate(order, start=1)
    }
    result["id_publico"] = result.index.map(identifiers)
    return result


def _percentage(series: pd.Series, value: str) -> float | None:
    return None if series.empty else float(series.eq(value).mean() * 100)


def _portfolio(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize capex strictly within each edition, never across editions."""
    condition_counts = (
        panel.assign(n=1)
        .pivot_table(
            index="edicion", columns="condicion", values="n", aggfunc="sum", fill_value=0
        )
        .reindex(columns=["BASE", "PROBABLE", "POSIBLE", "POTENCIAL"], fill_value=0)
    )
    totals = panel.groupby("edicion", as_index=True).agg(
        n_proyectos=("project_id", "nunique"),
        capex_nominal_total=("inversion_mmusd", "sum"),
        capex_real_total=("capex_real_2025", "sum"),
    )
    result = totals.join(condition_counts).reset_index()
    return result.rename(
        columns={
            "BASE": "n_base",
            "PROBABLE": "n_probable",
            "POSIBLE": "n_posible",
            "POTENCIAL": "n_potencial",
        }
    )


def _drift_distribution(transitions: pd.DataFrame) -> pd.DataFrame:
    source = transitions.dropna(subset=["delta_pem", "condicion_prev"]).copy()
    source["clase_delta"] = source["delta_pem"].map(delta_class)
    result = (
        source.groupby(["condicion_prev", "clase_delta"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    result["clase_delta"] = pd.Categorical(
        result["clase_delta"], categories=DELTA_CLASSES, ordered=True
    )
    return result.sort_values(["condicion_prev", "clase_delta"]).reset_index(drop=True)


def _delay_by_condition(transitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, group in transitions.dropna(subset=["delta_pem"]).groupby(
        "condicion_prev", dropna=False
    ):
        values = pd.to_numeric(group["delta_pem"], errors="coerce").dropna()
        rows.append(
            {
                "condicion_prev": condition,
                "n_transiciones": len(values),
                "pct_atraso": float(values.gt(0).mean() * 100),
                "pct_sin_cambio": float(values.eq(0).mean() * 100),
                "pct_adelanto": float(values.lt(0).mean() * 100),
                "mediana_delta": float(values.median()),
                "p90_delta": float(values.quantile(0.9)),
            }
        )
    return pd.DataFrame(rows)


def _profile(projects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension in ("sector", "region", "condicion_inicial"):
        for category, group in projects.groupby(dimension, dropna=False):
            outcome = group["desenlace"]
            rows.append(
                {
                    "dimension": dimension,
                    "categoria": category,
                    "n_proyectos": len(group),
                    "mediana_capex_inicial_real": float(
                        pd.to_numeric(group["capex_inicial_real"], errors="coerce").median()
                    ),
                    "pct_operacion": _percentage(outcome, "entro_en_operacion"),
                    "pct_sin_operar": _percentage(outcome, "salio_sin_operar"),
                    "pct_censurado": _percentage(outcome, "censurado"),
                    "pct_vigente": _percentage(outcome, "vigente"),
                }
            )
    return pd.DataFrame(rows)


def build_descriptive_tables(
    panel: pd.DataFrame,
    projects: pd.DataFrame,
    transitions: pd.DataFrame,
    project_drift: pd.DataFrame,
    reconciliation: pd.DataFrame,
    cpi_path: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Build descriptive outputs after removing aggregate portfolio rows."""
    filtered = panel.loc[~panel["nombre_norm"].map(_is_aggregate)].copy()
    cpi = load_annual_cpi(cpi_path)
    missing = sorted(({2025} | set(filtered["edicion"].astype(int))) - set(cpi))
    if missing:
        raise ValueError(f"faltan valores CPI para: {', '.join(map(str, missing))}")
    filtered["capex_real_2025"] = pd.to_numeric(
        filtered["inversion_mmusd"], errors="coerce"
    ) * filtered["edicion"].map(lambda year: cpi[2025] / cpi[int(year)])
    projects_with_ids = add_public_ids(projects)
    id_map = projects_with_ids.set_index("project_id")["id_publico"]
    trajectories = filtered.assign(id_publico=filtered["project_id"].map(id_map)).rename(
        columns={
            "puesta_en_marcha": "pem",
            "capex_real_2025": "capex_real",
        }
    )[["id_publico", "edicion", "pem", "capex_real", "condicion"]]
    cohort = (
        project_drift.groupby(["primera_edicion", "desenlace"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    if reconciliation.empty:
        rotation = pd.DataFrame(
            columns=["edicion_desde", "edicion_hasta", "categoria", "n", "capex"]
        )
    else:
        rotation = reconciliation.rename(
            columns={"obs_n": "n", "obs_capex": "capex"}
        )[["edicion_desde", "edicion_hasta", "categoria", "n", "capex"]].copy()
    tables = {
        "desc_cartera_edicion": _portfolio(filtered),
        "desc_deriva_distribucion": _drift_distribution(transitions),
        "desc_atraso_condicion": _delay_by_condition(transitions),
        "desc_desenlace_cohorte": cohort,
        "desc_rotacion": rotation,
        "desc_perfil": _profile(project_drift),
        "trayectorias": trajectories,
    }
    return tables, projects_with_ids


def write_descriptive_tables(
    tables: dict[str, pd.DataFrame], interim_dir: Path
) -> None:
    """Write every descriptive table as parquet."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(interim_dir / f"{name}.parquet", index=False)
