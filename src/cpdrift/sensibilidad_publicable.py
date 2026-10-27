"""Publication-ready outcome and threshold sensitivity scenarios."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .indicators import _build_transitions, _with_real_capex, load_annual_cpi
from .match import assign_project_ids
from .panel import build_projects
from .regression import fit_ols
from .survival import build_cif_table

METRICS = (
    "cif_op_6_base_probable",
    "cif_sal_6_base_probable",
    "cif_op_6_posible_potencial",
    "cif_sal_6_posible_potencial",
    "ols_coef_potencial",
)
POTENTIAL_TERM = 'C(condicion_prev, Treatment("BASE"))[T.POTENCIAL]'
DESCRIPTIONS = {
    "principal": "Especificación principal sin cambios.",
    "censurados_como_salida": "Todos los desenlaces censurados se tratan como salida sin operar.",
    "censurados_como_operacion": "Todos los desenlaces censurados se tratan como entrada en operación.",
    "cuadre_como_operacion": "Los egresos respaldados por cuadro se tratan como operación.",
    "cuadre_como_salida": "Los egresos respaldados por cuadro se tratan como salida sin operar.",
    "sin_vinculos_debiles": (
        "Los vínculos manuales por operador/capex o conciliación se rompen; "
        "el predecesor queda censurado y no se aplica la compuerta de conciliación."
    ),
    "capex_inicial_ge_90": "Solo proyectos con capex nominal inicial de al menos 90 MMUS$.",
}


def _endpoint_projects(panel: pd.DataFrame, projects: pd.DataFrame) -> pd.DataFrame:
    first = (
        panel.sort_values("edicion", kind="stable")
        .drop_duplicates("project_id")
        .set_index("project_id")
    )
    result = projects.copy()
    result["condicion_inicial"] = result["project_id"].map(first["condicion"])
    result["capex_inicial_nom"] = result["project_id"].map(first["inversion_mmusd"])
    return result


def _metrics(
    scenario: str,
    panel: pd.DataFrame,
    projects: pd.DataFrame,
    transitions: pd.DataFrame,
) -> dict[str, object]:
    cif = build_cif_table(projects, bootstrap_replicates=0)

    def cif_value(group: str, column: str) -> float:
        row = cif.loc[
            cif["cohorte"].eq("incidente")
            & cif["grupo"].eq(group)
            & cif["t"].eq(6)
        ]
        if row.empty:
            raise ValueError(f"CIF ausente en escenario {scenario}: {group}")
        return float(row.iloc[0][column])

    ols = fit_ols(transitions)
    coefficient = ols.loc[ols["termino"].eq(POTENTIAL_TERM), "coef"]
    if len(coefficient) != 1:
        raise ValueError(f"coeficiente Potencial ausente en escenario {scenario}")
    return {
        "escenario": scenario,
        "descripcion": DESCRIPTIONS[scenario],
        "cif_op_6_base_probable": cif_value("BASE_PROBABLE", "cif_op"),
        "cif_sal_6_base_probable": cif_value("BASE_PROBABLE", "cif_sal"),
        "cif_op_6_posible_potencial": cif_value("POSIBLE_POTENCIAL", "cif_op"),
        "cif_sal_6_posible_potencial": cif_value("POSIBLE_POTENCIAL", "cif_sal"),
        "ols_coef_potencial": float(coefficient.iloc[0]),
        "n_proyectos": int(projects["project_id"].nunique()),
        "n_transiciones": len(transitions),
    }


def _weak_panel_and_projects(
    normalized: pd.DataFrame,
    egresos: pd.DataFrame,
    manual_path: Path,
    interim_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel, _, _ = assign_project_ids(
        normalized,
        manual_path,
        interim_dir / "revision_cruce_sin_vinculos_debiles.csv",
        weak_evidence=frozenset({"operador_capex", "conciliacion"}),
    )
    weak_egresos = egresos.copy()
    existing = set(
        zip(
            weak_egresos["edicion_desde"].astype(int),
            weak_egresos["proyecto_raw"].astype(str),
            strict=True,
        )
    )
    additions = []
    for _, appearances in panel.groupby("project_id", sort=False):
        last = appearances.sort_values("edicion", kind="stable").iloc[-1]
        edition = int(last["edicion"])
        key = (edition, str(last["proyecto_raw"]))
        if edition == 2025 or key in existing:
            continue
        additions.append(
            {
                "edicion_desde": edition,
                "edicion_hasta": edition + 1,
                "proyecto_raw": last["proyecto_raw"],
                "motivo": "no_declarado",
                "fuente": "sensibilidad",
                "pagina": "",
                "frase": "",
                "nota": "Vínculo débil roto para sensibilidad.",
            }
        )
    if additions:
        weak_egresos = pd.concat(
            [weak_egresos, pd.DataFrame(additions)], ignore_index=True
        )
    projects = build_projects(panel, weak_egresos)
    panel.to_parquet(interim_dir / "panel_sin_vinculos_debiles.parquet", index=False)
    return panel, projects


def build_publishable_sensitivity(
    panel: pd.DataFrame,
    projects: pd.DataFrame,
    transitions: pd.DataFrame,
    normalized: pd.DataFrame,
    egresos: pd.DataFrame,
    cpi_path: Path,
    manual_path: Path,
    interim_dir: Path,
) -> pd.DataFrame:
    """Calculate the seven prespecified publication sensitivity scenarios."""
    endpoints = _endpoint_projects(panel, projects)
    scenarios: list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = [
        ("principal", panel, endpoints, transitions)
    ]
    for name, outcome in (
        ("censurados_como_salida", "salio_sin_operar"),
        ("censurados_como_operacion", "entro_en_operacion"),
    ):
        changed = endpoints.copy()
        changed.loc[changed["desenlace"].eq("censurado"), "desenlace"] = outcome
        scenarios.append((name, panel, changed, transitions))
    for name, outcome in (
        ("cuadre_como_operacion", "entro_en_operacion"),
        ("cuadre_como_salida", "salio_sin_operar"),
    ):
        changed = endpoints.copy()
        changed.loc[changed["fuente_desenlace"].eq("cuadro"), "desenlace"] = outcome
        scenarios.append((name, panel, changed, transitions))
    weak_panel, weak_projects = _weak_panel_and_projects(
        normalized, egresos, manual_path, interim_dir
    )
    weak_real, _ = _with_real_capex(weak_panel, load_annual_cpi(cpi_path))
    weak_transitions = _build_transitions(weak_real)
    weak_endpoints = _endpoint_projects(weak_panel, weak_projects)
    scenarios.append(
        (
            "sin_vinculos_debiles",
            weak_panel,
            weak_endpoints,
            weak_transitions,
        )
    )
    eligible = set(
        endpoints.loc[
            pd.to_numeric(endpoints["capex_inicial_nom"], errors="coerce").ge(90),
            "project_id",
        ]
    )
    scenarios.append(
        (
            "capex_inicial_ge_90",
            panel.loc[panel["project_id"].isin(eligible)],
            endpoints.loc[endpoints["project_id"].isin(eligible)],
            transitions.loc[transitions["project_id"].isin(eligible)],
        )
    )
    return pd.DataFrame(
        [_metrics(name, scenario_panel, scenario_projects, scenario_transitions)
         for name, scenario_panel, scenario_projects, scenario_transitions in scenarios]
    )


def write_publishable_sensitivity(
    table: pd.DataFrame, interim_dir: Path, powerbi_dir: Path
) -> tuple[Path, Path]:
    """Write the exact parquet and CSV publication artifacts."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    powerbi_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = interim_dir / "sensibilidad_publicable.parquet"
    csv_path = powerbi_dir / "sens_publicable.csv"
    table.to_parquet(parquet_path, index=False)
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return parquet_path, csv_path
