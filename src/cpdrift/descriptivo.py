"""Publication-ready descriptive tables for the longitudinal portfolio."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .etiquetas import CATEGORY_LABELS, GROUP_LABELS
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
    trajectories = enrich_trajectories(trajectories)
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


def enrich_trajectories(table: pd.DataFrame) -> pd.DataFrame:
    """Add time and launch-year drift relative to each project's first row."""
    result = table.sort_values(["id_publico", "edicion"], kind="stable").copy()
    first_edition = result.groupby("id_publico")["edicion"].transform(
        lambda values: values.iloc[0]
    )
    first_pem = result.groupby("id_publico")["pem"].transform(
        lambda values: values.iloc[0]
    )
    result["anios_desde_primera"] = result["edicion"] - first_edition
    result["deriva_desde_primera"] = result["pem"] - first_pem
    return result


def build_fan_table(trajectories: pd.DataFrame) -> pd.DataFrame:
    """Summarize the distribution of accumulated drift by observed distance."""
    rows = []
    for distance, group in trajectories.dropna(subset=["deriva_desde_primera"]).groupby("anios_desde_primera"):
        values = group["deriva_desde_primera"].to_numpy(dtype=float)
        projects = int(group["id_publico"].nunique())
        if projects < 10:
            continue
        percentiles = np.percentile(values, [10, 25, 50, 75, 90])
        rows.append({"anios_desde_primera": distance, "n_proyectos": projects, "p10": percentiles[0], "p25": percentiles[1], "mediana": percentiles[2], "p75": percentiles[3], "p90": percentiles[4]})
    return pd.DataFrame(rows, columns=["anios_desde_primera", "n_proyectos", "p10", "p25", "mediana", "p75", "p90"])


def build_ten_table(cif: pd.DataFrame) -> pd.DataFrame:
    """Express incident competing risks as counts out of ten projects."""
    selected = cif.loc[cif["cohorte"].eq("incidente") & cif["grupo"].isin(GROUP_LABELS) & cif["t"].isin({4, 6})]
    rows = []
    for row in selected.itertuples(index=False):
        operation = round(10 * float(row.cif_op))
        exits = round(10 * float(row.cif_sal))
        remainder = 10 - operation - exits
        if remainder < 0:
            raise ValueError("resto negativo en visualización de cada 10")
        rows.append({"grupo": row.grupo, "grupo_etiqueta": GROUP_LABELS[row.grupo], "horizonte": int(row.t), "operan": operation, "salen": exits, "resto": remainder})
    return pd.DataFrame(rows)


def build_sensitivity_potential(table: pd.DataFrame) -> pd.DataFrame:
    """Pivot the expanded OLS potential term for the three regression scenarios."""
    labels = {"principal": "Principal", "sin_par_2021_2023": "Sin el par 2021→2023", "sin_estatales": "Sin Codelco ni Enami"}
    term = 'C(condicion_prev, Treatment("BASE"))[T.POTENCIAL]'
    selected = table.loc[table["modelo"].eq("ols_ampliado") & table["metrica"].isin({f"coef:{term}", f"p:{term}"})].copy()
    selected["tipo"] = selected["metrica"].str.split(":", n=1).str[0]
    result = selected.pivot(index="escenario", columns="tipo", values="valor").reset_index()
    result.insert(1, "escenario_etiqueta", result["escenario"].map(labels))
    if result["escenario_etiqueta"].isna().any():
        raise ValueError("escenario sin etiqueta")
    return result[["escenario", "escenario_etiqueta", "coef", "p"]]


def build_reconciliation_matrix(table: pd.DataFrame) -> pd.DataFrame:
    """Build the ordered compact reconciliation matrix."""
    result = table[["edicion_desde", "edicion_hasta", "categoria", "estado"]].copy()
    result["par"] = result["edicion_desde"].astype(str) + "→" + result["edicion_hasta"].astype(str)
    pairs = result[["edicion_desde", "edicion_hasta"]].drop_duplicates().sort_values(["edicion_desde", "edicion_hasta"])
    order = {(row.edicion_desde, row.edicion_hasta): index for index, row in enumerate(pairs.itertuples(index=False), start=1)}
    result["orden_par"] = [order[(row.edicion_desde, row.edicion_hasta)] for row in result.itertuples(index=False)]
    result["categoria_etiqueta"] = result["categoria"].map(CATEGORY_LABELS)
    if result["categoria_etiqueta"].isna().any():
        raise ValueError("categoría sin etiqueta")
    result["simbolo"] = result["estado"].map(lambda value: "✓" if value == "ok" else "")
    output = result[["par", "orden_par", "categoria_etiqueta", "estado", "simbolo"]]
    if len(output) != 30:
        raise ValueError(f"matriz de conciliación requiere 30 filas, recibió {len(output)}")
    return output


def write_visual_tables(tables: dict[str, pd.DataFrame], interim_dir: Path) -> None:
    """Write dashboard-specific visual tables as parquet."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(interim_dir / f"{name}.parquet", index=False)
