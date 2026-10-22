"""Build the Sprint 2 JSON evidence and matching Markdown report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

DISCLAIMER = "EXPLORATORIO — no publicable"
DECISIONES = [
    {"decision": "Se conserva evidencia literal y página para cada cifra declarada."},
    {"decision": "Los cruces dudosos se derivan a revisión manual sin reutilizar identidad."},
    {"decision": "Las imputaciones regionales usan solo apariciones del mismo proyecto."},
]


def build_report(
    editions: list[dict[str, Any]],
    identity: list[dict[str, Any]],
    projects: list[dict[str, Any]],
    unknowns: dict[str, list[str]],
    decisions: list[dict[str, Any]] | None = None,
    indicators: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    outcomes: dict[str, int] = {}
    reentries = 0
    for project in projects:
        outcome = str(project["desenlace"])
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        reentries += int(bool(project["reingreso"]))
    return {
        "encabezado": DISCLAIMER,
        "ediciones": editions,
        "identidad": identity,
        "desenlaces": {"categorias": outcomes, "reingresos": reentries},
        "catalogos": unknowns,
        "indicadores": [] if indicators is None else indicators,
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
    json_path = interim_dir / "sprint2_reporte.json"
    md_path = interim_dir / "sprint2_reporte.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
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
    indicators = _table(
        ["Cifra", "Valor", "Unidad", "Estado"],
        [
            [item["descripcion"], item["valor"], item["unidad"], item["estado"]]
            for item in report["indicadores"]
        ],
    )
    text = (
        f"# {DISCLAIMER}\n\n## Conciliación por edición\n\n{reconciliation}\n\n"
        f"## Identidad\n\n{identity}\n\n## Desenlaces\n\n"
        + "\n".join(
            f"- {key}: {value}" for key, value in outcomes["categorias"].items()
        )
        + f"\n- reingresos: {outcomes['reingresos']}\n\n## Indicadores\n\n"
        + indicators
        + "\n\n## Catálogos desconocidos\n\n"
        + "\n".join(f"- {key}: {values}" for key, values in report["catalogos"].items())
        + "\n\n## Decisiones tomadas\n\n"
        + "\n".join(f"- {item}" for item in report["decisiones_tomadas"])
        + "\n"
    )
    md_path.write_text(text, encoding="utf-8")
    return md_path, json_path
