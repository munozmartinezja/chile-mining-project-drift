"""Longitudinal project-level derivations and parquet outputs."""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

DATA_EDITIONS = [*range(2014, 2022), 2023, 2024, 2025]
AGGREGATE_PREFIX = "OTROS"
AGGREGATE_CONTAINS = "PROYECTOS DE INFORMACION"
AGGREGATE_EXACT_NAMES = frozenset({"PROYECTOS FURE", "TRANQUES"})
OUTCOME_BY_REASON = {
    "operacion": "entro_en_operacion",
    "reestructuracion": "salio_sin_operar",
    "metodologia": "censurado",
    "fusion": "censurado",
    "no_declarado": "censurado",
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


def build_projects(panel: pd.DataFrame, egresos: pd.DataFrame) -> pd.DataFrame:
    """Collapse appearances to one lifecycle row per project ID."""
    records: list[dict[str, object]] = []
    edition_positions = {edition: index for index, edition in enumerate(DATA_EDITIONS)}
    eligible = panel.loc[~panel["nombre_norm"].map(_is_aggregate)]
    for project_id, appearances in eligible.groupby("project_id", sort=False):
        group = appearances.sort_values("edicion", kind="stable")
        editions = [int(value) for value in group["edicion"]]
        first = group.iloc[0]
        last = group.iloc[-1]
        gaps = any(
            edition_positions[right] - edition_positions[left] > 1
            for left, right in itertools.pairwise(editions)
        )
        if 2025 in editions:
            outcome = "vigente"
            reason = ""
            source = "vigente"
        else:
            matches = egresos.loc[
                egresos["edicion_desde"].eq(editions[-1])
                & egresos["proyecto_raw"].eq(last["proyecto_raw"])
            ]
            if len(matches) != 1:
                raise ValueError(
                    "egreso no único para "
                    f"{editions[-1]} / {last['proyecto_raw']}: {len(matches)}"
                )
            exit_row = matches.iloc[0]
            reason = str(exit_row["motivo"])
            if reason not in OUTCOME_BY_REASON:
                raise ValueError(f"motivo de salida desconocido: {reason}")
            outcome = OUTCOME_BY_REASON[reason]
            source = str(exit_row["fuente"])
        records.append(
            {
                "project_id": project_id,
                "primera_edicion": editions[0],
                "ultima_edicion": editions[-1],
                "n_ediciones": len(editions),
                "ediciones": editions,
                "pem_primera": first["puesta_en_marcha"],
                "pem_ultima": last["puesta_en_marcha"],
                "capex_primero": first["inversion_mmusd"],
                "capex_ultimo": last["inversion_mmusd"],
                "desenlace": outcome,
                "motivo_salida": reason,
                "fuente_desenlace": source,
                "reingreso": gaps,
            }
        )
    return pd.DataFrame(records)


def write_outputs(
    panel: pd.DataFrame,
    interim_dir: Path,
    egresos: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interim_dir.mkdir(parents=True, exist_ok=True)
    projects = build_projects(panel, egresos)
    panel.to_parquet(interim_dir / "panel.parquet", index=False)
    projects.to_parquet(interim_dir / "proyectos.parquet", index=False)
    return panel, projects
