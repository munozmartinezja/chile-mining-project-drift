"""Longitudinal project-level derivations and parquet outputs."""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

DATA_EDITIONS = [*range(2014, 2022), 2023, 2024, 2025]


def build_projects(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse appearances to one lifecycle row per project ID."""
    records: list[dict[str, object]] = []
    edition_positions = {edition: index for index, edition in enumerate(DATA_EDITIONS)}
    for project_id, appearances in panel.groupby("project_id", sort=False):
        group = appearances.sort_values("edicion", kind="stable")
        editions = [int(value) for value in group["edicion"]]
        first = group.iloc[0]
        last = group.iloc[-1]
        gaps = any(
            edition_positions[right] - edition_positions[left] > 1
            for left, right in itertools.pairwise(editions)
        )
        pem_last = last["puesta_en_marcha"]
        if 2025 in editions:
            outcome = "vigente"
        elif pd.notna(pem_last) and int(pem_last) <= editions[-1] + 1:
            outcome = "entro_en_operacion"
        else:
            outcome = "salio_sin_operar"
        records.append(
            {
                "project_id": project_id,
                "primera_edicion": editions[0],
                "ultima_edicion": editions[-1],
                "n_ediciones": len(editions),
                "ediciones": editions,
                "pem_primera": first["puesta_en_marcha"],
                "pem_ultima": pem_last,
                "capex_primero": first["inversion_mmusd"],
                "capex_ultimo": last["inversion_mmusd"],
                "desenlace": outcome,
                "reingreso": gaps,
            }
        )
    return pd.DataFrame(records)


def write_outputs(panel: pd.DataFrame, interim_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    interim_dir.mkdir(parents=True, exist_ok=True)
    projects = build_projects(panel)
    panel.to_parquet(interim_dir / "panel.parquet", index=False)
    projects.to_parquet(interim_dir / "proyectos.parquet", index=False)
    return panel, projects
