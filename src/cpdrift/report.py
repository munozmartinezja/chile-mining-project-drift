"""Build the diagnostic JSON evidence and matching Markdown report."""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd

DISCLAIMER = "Informe de diagnóstico del pipeline. Las cifras publicables están en docs/registro_cifras.csv."
DECISIONES = [
    {"decision": "Se conserva evidencia literal y página para cada cifra declarada."},
    {"decision": "Los cruces dudosos se derivan a revisión manual sin reutilizar identidad."},
    {"decision": "Las imputaciones regionales usan solo apariciones del mismo proyecto."},
]


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Real) and not math.isfinite(float(value)):
        return None
    if not isinstance(value, (str, bool)) and value is not None and pd.isna(value):
        return None
    return value


def build_report(
    editions: list[dict[str, Any]],
    identity: list[dict[str, Any]],
    projects: list[dict[str, Any]],
    unknowns: dict[str, list[str]],
    decisions: list[dict[str, Any]] | None = None,
    indicators: list[dict[str, Any]] | None = None,
    models: dict[str, list[dict[str, Any]]] | None = None,
    outcome_reconciliation: list[dict[str, Any]] | None = None,
    robustness: list[dict[str, Any]] | None = None,
    temporal_validation: list[dict[str, Any]] | None = None,
    aggregate_validation: list[dict[str, Any]] | None = None,
    single_cut_validation: list[dict[str, Any]] | None = None,
    validation_intervals: dict[str, float] | None = None,
    descriptive: dict[str, list[dict[str, Any]]] | None = None,
    calibration: list[dict[str, Any]] | None = None,
    date_rule: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    outcomes: dict[str, int] = {}
    outcome_sources: dict[str, int] = {}
    reentries = 0
    for project in projects:
        outcome = str(project["desenlace"])
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        source = str(project.get("fuente_desenlace", ""))
        outcome_sources[source] = outcome_sources.get(source, 0) + 1
        reentries += int(bool(project["reingreso"]))
    return {
        "encabezado": DISCLAIMER,
        "ediciones": editions,
        "identidad": identity,
        "desenlaces": {
            "categorias": outcomes,
            "fuentes": outcome_sources,
            "reingresos": reentries,
        },
        "conciliacion_desenlaces": (
            [] if outcome_reconciliation is None else outcome_reconciliation
        ),
        "catalogos": unknowns,
        "indicadores": [] if indicators is None else indicators,
        "modelos": {"cif": [], "ols": [], "logit": []} if models is None else models,
        "robustez": [] if robustness is None else robustness,
        "validacion_temporal": (
            [] if temporal_validation is None else temporal_validation
        ),
        "validacion_agregada": (
            [] if aggregate_validation is None else aggregate_validation
        ),
        "validacion_corte_unico": (
            [] if single_cut_validation is None else single_cut_validation
        ),
        "intervalos_validacion": (
            {} if validation_intervals is None else validation_intervals
        ),
        "descriptivo": {} if descriptive is None else descriptive,
        "calibracion_test": [] if calibration is None else calibration,
        "regla_fecha": [] if date_rule is None else date_rule,
        "decisiones_tomadas": DECISIONES if decisions is None else decisions,
    }


def _table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join("—" if pd.isna(value) else str(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], interim_dir: Path) -> tuple[Path, Path]:
    interim_dir.mkdir(parents=True, exist_ok=True)
    json_path = interim_dir / "diagnostico.json"
    md_path = interim_dir / "diagnostico.md"
    json_path.write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    reconciliation = _table(
        ["Edición", "Filas", "Declarados", "Suma", "Total", "Estado", "Umbral"],
        [
            [
                item["edicion"], item["filas"], item["declarados"], item["suma_mmusd"],
                item["total_mmusd"], item["estado"], item["umbral_edicion"],
            ]
            for item in report["ediciones"]
        ],
    )
    identity = _table(
        ["Edición", "% automático", "Ambiguos", "Nuevos", "Manuales"],
        [
            [
                item["edicion"], item["automatico_pct"], item["ambiguos"],
                item["nuevos"], item["manuales"],
            ]
            for item in report["identidad"]
        ],
    )
    outcomes = report["desenlaces"]
    outcome_reconciliation = _table(
        [
            "Desde",
            "Hasta",
            "Categoría",
            "N declarado",
            "N observado",
            "Capex declarado",
            "Capex observado",
            "Delta N",
            "Delta capex",
            "Compara capex",
            "Estado",
        ],
        [
            [
                item["edicion_desde"],
                item["edicion_hasta"],
                item["categoria"],
                item["decl_n"],
                item["obs_n"],
                item["decl_capex"],
                item["obs_capex"],
                item["delta_n"],
                item["delta_capex"],
                item["compara_capex"],
                item["estado"],
            ]
            for item in report["conciliacion_desenlaces"]
        ],
    )
    indicators = _table(
        ["Cifra", "Valor", "Unidad", "Estado"],
        [
            [item["descripcion"], item["valor"], item["unidad"], item["estado"]]
            for item in report["indicadores"]
            if item["estado"] != "informativa"
        ],
    )
    models = report["modelos"]
    cif = _table(
        ["Cohorte", "Grupo", "t", "CIF operación", "CIF salida"],
        [
            [
                item["cohorte"],
                item["grupo"],
                item["t"],
                item["cif_op"],
                item["cif_sal"],
            ]
            for item in models["cif"]
            if item["t"] in {2, 4, 6}
        ],
    )
    ols = _table(
        ["Término", "Coeficiente", "EE", "IC inferior", "IC superior", "p", "n", "R²"],
        [
            [
                item["termino"], item["coef"], item["ee"], item["ic_inf"],
                item["ic_sup"], item["p"], item["n"], item["r2"],
            ]
            for item in models["ols"]
        ],
    )
    logit = _table(
        ["Término", "Razón de odds", "EE", "Tipo EE", "IC inferior", "IC superior", "p", "n", "Pseudo-R²"],
        [
            [
                item["termino"], item["coef"], item["ee"], item.get("ee_tipo", ""), item["ic_inf"],
                item["ic_sup"], item["p"], item["n"], item["pseudo_r2"],
            ]
            for item in models["logit"]
        ],
    )
    ols_extended = _table(
        ["Término", "Coeficiente", "EE", "IC inferior", "IC superior", "p", "n", "R²"],
        [
            [
                item["termino"], item["coef"], item["ee"], item["ic_inf"],
                item["ic_sup"], item["p"], item["n"], item["r2"],
            ]
            for item in models.get("ols_ampliado", [])
        ],
    )
    robustness = _table(
        ["Escenario", "Modelo", "Métrica", "Valor", "n"],
        [
            [item["escenario"], item["modelo"], item["metrica"], item["valor"], item["n"]]
            for item in report["robustez"]
        ],
    )
    validation = _table(
        ["Corte", "Modelo", "AUC", "Brier", "Calibración", "n train", "n test"],
        [
            [item["corte"], item["modelo"], item["auc"], item["brier"], item["calibracion"], item["n_train"], item["n_test"]]
            for item in report["validacion_temporal"]
        ],
    )
    aggregate_validation = _table(
        ["Modelo", "AUC", "Brier", "Calibración", "n test"],
        [
            [item["modelo"], item["auc"], item["brier"], item["calibracion"], item["n_test"]]
            for item in report["validacion_agregada"]
        ],
    )
    single_cut = _table(
        ["Modelo", "AUC", "Brier", "Calibración", "n train", "n test"],
        [
            [item["modelo"], item["auc"], item["brier"], item["calibracion_global"], item["n_train"], item["n_test"]]
            for item in report["validacion_corte_unico"]
        ],
    )
    calibration = _table(
        ["Quintil", "Media predicha", "Tasa observada", "n"],
        [
            [item["quintil"], item["media_predicha"], item["tasa_observada"], item["n"]]
            for item in report["calibracion_test"]
        ],
    )
    date_rule = _table(
        ["Declarado", "Regla", "n"],
        [[item["declarado"], item["regla"], item["n"]] for item in report["regla_fecha"]],
    )
    portfolio = _table(
        ["Edición", "Proyectos", "Capex nominal", "Capex real", "Base", "Probable", "Posible", "Potencial"],
        [
            [item["edicion"], item["n_proyectos"], item["capex_nominal_total"], item["capex_real_total"], item["n_base"], item["n_probable"], item["n_posible"], item["n_potencial"]]
            for item in report["descriptivo"].get("desc_cartera_edicion", [])
        ],
    )
    delay = _table(
        ["Condición", "Transiciones", "% atraso", "% sin cambio", "% adelanto", "Mediana", "p90"],
        [
            [item["condicion_prev"], item["n_transiciones"], item["pct_atraso"], item["pct_sin_cambio"], item["pct_adelanto"], item["mediana_delta"], item["p90_delta"]]
            for item in report["descriptivo"].get("desc_atraso_condicion", [])
        ],
    )
    text = (
        f"# {DISCLAIMER}\n\n## Conciliación por edición\n\n{reconciliation}\n\n"
        f"## Identidad\n\n{identity}\n\n## Desenlaces\n\n"
        + "\n".join(
            f"- {key}: {value}" for key, value in outcomes["categorias"].items()
        )
        + f"\n- reingresos: {outcomes['reingresos']}\n\n"
        + "### Fuentes de desenlace\n\n"
        + "\n".join(
            f"- {key}: {value}" for key, value in outcomes["fuentes"].items()
        )
        + "\n\n## Conciliación de desenlaces\n\n"
        + outcome_reconciliation
        + "\n\n## Indicadores\n\n"
        + indicators
        + "\n\n## Descriptivo\n\n### Cartera por edición\n\n"
        + portfolio
        + "\n\n### Deriva por condición\n\n"
        + delay
        + "\n\n## Modelos\n\nresultados asociativos.\n\n"
        + "### Incidencia acumulada\n\n"
        + cif
        + "\n\n### OLS\n\n"
        + ols
        + "\n\n### Logit\n\n"
        + logit
        + "\n\n### OLS ampliado\n\n"
        + ols_extended
        + "\n\n## Robustez\n\n"
        + robustness
        + "\n\n## Validación temporal\n\n"
        + validation
        + "\n\n### Resultados agregados\n\n"
        + aggregate_validation
        + "\n\nIntervalos bootstrap agregados: "
        + str(report["intervalos_validacion"])
        + "\n\n### Calibración por quintiles\n\n"
        + calibration
        + "\n\n### Sensibilidad: corte único\n\n"
        + single_cut
        + "\n\n## Regla de fecha contra desenlace declarado\n\n"
        + date_rule
        + "\n\n## Catálogos desconocidos\n\n"
        + "\n".join(f"- {key}: {values}" for key, values in report["catalogos"].items())
        + "\n\n## Decisiones tomadas\n\n"
        + "\n".join(f"- {item}" for item in report["decisiones_tomadas"])
        + "\n"
    )
    md_path.write_text(text, encoding="utf-8")
    return md_path, json_path
