"""Anonymous CSV exports for the publication data model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COPPER_SECTORS = {"GRAN_MINERIA_CU", "ESTATAL_CU", "MEDIANA_MINERIA_CU"}


def _sector_group(value: object) -> str:
    if value in COPPER_SECTORS:
        return "COBRE"
    if str(value).upper() == "ORO":
        return "ORO"
    return "OTROS"


def _write(table: pd.DataFrame, path: Path) -> Path:
    table.to_csv(path, index=False, encoding="utf-8-sig", sep=",", decimal=".")
    return path


def _ensure_anonymous(table: pd.DataFrame, raw_names: set[str], name: str) -> None:
    prohibited = [
        column
        for column in table.columns
        if column == "project_id"
        or column == "proyecto_raw"
        or column.startswith(("operador_", "nombre_"))
    ]
    if prohibited:
        raise ValueError(f"columnas identificadoras en {name}: {', '.join(prohibited)}")
    text_values = {
        str(value)
        for value in table.select_dtypes(include=["object", "string"]).to_numpy().ravel()
        if pd.notna(value)
    }
    matches = sorted(raw_names & text_values)
    if matches:
        raise ValueError(f"texto identificador en {name}: {matches[0]}")


def export_powerbi(
    output_dir: Path,
    *,
    panel: pd.DataFrame,
    projects: pd.DataFrame,
    transitions: pd.DataFrame,
    descriptive: dict[str, pd.DataFrame],
    cif: pd.DataFrame,
    reconciliation: pd.DataFrame,
    validation_cuts: pd.DataFrame,
    validation_aggregate: pd.DataFrame,
    aggregate_calibration: pd.DataFrame,
    sensitivity: pd.DataFrame,
    date_rule_matrix: pd.DataFrame,
    copper_prices: pd.DataFrame,
    claims: list[dict[str, object]],
) -> list[Path]:
    """Write the anonymous publication tables as comma-separated UTF-8 files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_names = set(panel.get("proyecto_raw", pd.Series(dtype=str)).dropna().astype(str))
    project_columns = [
        "id_publico",
        "sector",
        "region",
        "condicion_inicial",
        "primera_edicion",
        "ultima_edicion",
        "n_ediciones",
        "desenlace",
        "fuente_desenlace",
        "reingreso",
        "capex_inicial_real",
    ]
    dimension = projects[project_columns].copy()
    dimension.insert(2, "grupo_sector", dimension["sector"].map(_sector_group))
    identifier_map = projects.set_index("project_id")["id_publico"]
    if transitions.empty:
        fact_transition = pd.DataFrame(
            columns=[
                "id_publico", "ed_prev", "ed_next", "gap_anios", "delta_pem",
                "tasa_deriva", "atraso", "condicion_prev", "capex_prev_real",
                "delta_capex_real_pct", "capex_actualizado",
            ]
        )
    else:
        fact_transition = transitions.assign(
            id_publico=transitions["project_id"].map(identifier_map),
            atraso=pd.to_numeric(transitions["delta_pem"], errors="coerce").gt(0),
            capex_actualizado=transitions["capex_prev_nom"].ne(
                transitions["capex_next_nom"]
            ),
        )[
            [
                "id_publico", "ed_prev", "ed_next", "gap_anios", "delta_pem",
                "tasa_deriva", "atraso", "condicion_prev", "capex_prev_real",
                "delta_capex_real_pct", "capex_actualizado",
            ]
        ]
    register = pd.DataFrame(claims)
    register_columns = ["id", "descripcion", "valor", "unidad", "estado"]
    register = register.reindex(columns=register_columns)
    exports: dict[str, pd.DataFrame] = {
        "dim_proyecto": dimension,
        "fact_trayectoria": descriptive["trayectorias"],
        "fact_transicion": fact_transition,
        "cif": cif,
        "conciliacion_desenlaces": reconciliation,
        "validacion_cortes": validation_cuts,
        "validacion_agregada": validation_aggregate,
        "calibracion_agregada": aggregate_calibration,
        "sensibilidad": sensitivity,
        "regla_vs_declarado": date_rule_matrix,
        "precio_cobre_edicion": copper_prices,
        "registro_cifras": register,
    }
    exports.update(
        {name: table for name, table in descriptive.items() if name.startswith("desc_")}
    )
    paths = []
    for name, table in exports.items():
        _ensure_anonymous(table, raw_names, name)
        paths.append(_write(table, output_dir / f"{name}.csv"))
    return paths
