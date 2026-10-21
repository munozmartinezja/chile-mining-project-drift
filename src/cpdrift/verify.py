"""Independent recomputation of headline figures from primary output tables."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd

COUNT_IDS = {
    "n_proyectos",
    "n_filas_agregadas_excluidas",
    "n_transiciones",
    "n_salidas_sin_operar",
    "n_salidas_operacion",
}


def _aggregate_name(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    name = str(value)
    return name.startswith("OTROS") or "PROYECTOS DE INFORMACION" in name


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2)


def _percent(count: int, total: int) -> float | None:
    return None if total == 0 else count / total * 100


def _recalculate(
    panel_path: Path, projects_path: Path, cpi_path: Path
) -> dict[str, float | int | None]:
    panel = pd.read_parquet(panel_path)
    projects = pd.read_parquet(projects_path)
    cpi_frame = pd.read_csv(cpi_path)
    cpi = {
        int(row.anio): (float(row.s1) + float(row.s2)) / 2
        for row in cpi_frame.itertuples(index=False)
    }
    aggregate = panel["nombre_norm"].map(_aggregate_name)
    excluded = int(aggregate.sum())
    panel = panel.loc[~aggregate].copy()
    needed_years = {int(value) for value in panel["edicion"].unique()} | {2025}
    missing = sorted(needed_years - set(cpi))
    if missing:
        years = ", ".join(str(value) for value in missing)
        raise ValueError(f"faltan valores CPI para: {years}")
    panel["capex_real_2025"] = [
        (
            float(amount) * cpi[2025] / cpi[int(edition)]
            if pd.notna(amount)
            else float("nan")
        )
        for amount, edition in zip(
            panel["inversion_mmusd"], panel["edicion"], strict=True
        )
    ]

    deltas: list[float] = []
    rates: list[float] = []
    capex_changes: list[float] = []
    drift_by_project: dict[object, tuple[int, float | None]] = {}
    transition_count = 0
    for project_id, group in panel.groupby("project_id", sort=False):
        rows = [row for _, row in group.sort_values("edicion", kind="stable").iterrows()]
        first = rows[0]
        last = rows[-1]
        accumulated = None
        if pd.notna(first["puesta_en_marcha"]) and pd.notna(last["puesta_en_marcha"]):
            accumulated = float(last["puesta_en_marcha"]) - float(
                first["puesta_en_marcha"]
            )
        drift_by_project[project_id] = (len(rows), accumulated)
        for previous, following in itertools.pairwise(rows):
            transition_count += 1
            if pd.notna(previous["puesta_en_marcha"]) and pd.notna(
                following["puesta_en_marcha"]
            ):
                delta = float(following["puesta_en_marcha"]) - float(
                    previous["puesta_en_marcha"]
                )
                deltas.append(delta)
                rates.append(
                    delta
                    / (int(following["edicion"]) - int(previous["edicion"]))
                )
            prior_capex = previous["capex_real_2025"]
            next_capex = following["capex_real_2025"]
            if pd.notna(prior_capex) and pd.notna(next_capex) and float(prior_capex) != 0:
                capex_changes.append(
                    (float(next_capex) / float(prior_capex) - 1) * 100
                )

    project_outcomes = (
        projects.drop_duplicates("project_id").set_index("project_id")["desenlace"]
    )
    operated_drifts = [
        drift
        for project_id, (appearances, drift) in drift_by_project.items()
        if appearances >= 2
        and drift is not None
        and project_id in project_outcomes.index
        and project_outcomes.loc[project_id] == "entro_en_operacion"
    ]

    threshold = panel.loc[
        pd.to_numeric(panel["inversion_mmusd"], errors="coerce").ge(90)
    ]
    editions = sorted(int(value) for value in panel["edicion"].unique())
    edition_rows = {
        int(edition): group.drop_duplicates("project_id").set_index("project_id")
        for edition, group in threshold.groupby("edicion", sort=False)
    }
    exits_operation = 0
    exits_without_operation = 0
    for previous_edition, next_edition in itertools.pairwise(editions):
        previous = edition_rows.get(previous_edition, pd.DataFrame())
        following = edition_rows.get(next_edition, pd.DataFrame())
        previous_ids = set(previous.index)
        following_ids = set(following.index)
        for project_id in previous_ids - following_ids:
            pem = previous.loc[project_id, "puesta_en_marcha"]
            if pd.notna(pem) and float(pem) <= next_edition:
                exits_operation += 1
            else:
                exits_without_operation += 1
    return {
        "n_proyectos": int(panel["project_id"].nunique()),
        "n_filas_agregadas_excluidas": excluded,
        "n_transiciones": transition_count,
        "media_delta_pem": _mean(deltas),
        "mediana_delta_pem": _median(deltas),
        "pct_transiciones_atraso": _percent(sum(value > 0 for value in deltas), len(deltas)),
        "pct_transiciones_adelanto": _percent(sum(value < 0 for value in deltas), len(deltas)),
        "media_tasa_deriva": _mean(rates),
        "mediana_delta_capex_real_pct": _median(capex_changes),
        "pct_fiabilidad_a_tiempo": _percent(
            sum(value <= 0 for value in operated_drifts), len(operated_drifts)
        ),
        "pct_fiabilidad_1_anio": _percent(
            sum(value <= 1 for value in operated_drifts), len(operated_drifts)
        ),
        "mediana_deriva_acumulada_operados": _median(operated_drifts),
        "n_salidas_sin_operar": exits_without_operation,
        "n_salidas_operacion": exits_operation,
    }


def _matches(claim_id: str, stated: object, recalculated: object) -> bool:
    if stated is None or recalculated is None:
        return stated is None and recalculated is None
    if claim_id in COUNT_IDS:
        return float(stated) == float(recalculated)
    return abs(float(stated) - float(recalculated)) <= 1e-9


def verify_claims_register(
    claims: list[dict[str, object]],
    panel_path: Path,
    projects_path: Path,
    cpi_path: Path,
) -> list[dict[str, object]]:
    """Recalculate and label every claim without reading derived indicators."""
    expected = _recalculate(panel_path, projects_path, cpi_path)
    verified: list[dict[str, object]] = []
    for claim in claims:
        item = dict(claim)
        recalculated = expected.get(str(item["id"]))
        item["valor_recalculado"] = recalculated
        item["estado"] = (
            "verificada"
            if str(item["id"]) in expected
            and _matches(str(item["id"]), item["valor"], recalculated)
            else "fallida"
        )
        verified.append(item)
    return verified


def write_claims_register(claims: list[dict[str, object]], path: Path) -> None:
    """Write the verified register as strict JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(claims, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def require_verified(claims: list[dict[str, object]]) -> None:
    """Stop the pipeline when at least one registered figure differs."""
    failures = [str(item["id"]) for item in claims if item["estado"] == "fallida"]
    if failures:
        raise ValueError("cifras no verificadas: " + ", ".join(failures))
