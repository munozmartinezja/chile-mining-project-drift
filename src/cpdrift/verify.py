"""Independent recomputation of headline figures from primary output tables."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

COUNT_IDS = {
    "n_proyectos",
    "n_filas_agregadas_excluidas",
    "n_transiciones",
    "n_salidas_umbral90_sin_operar",
    "n_salidas_umbral90_operacion",
    "n_desenlace_operacion",
    "n_desenlace_sin_operar",
    "n_desenlace_censurado",
    "n_pares_conciliados",
    "n_cohorte_incidente",
    "ols_n",
}

DATA_EDITIONS = [*range(2014, 2022), 2023, 2024, 2025]
AGGREGATE_PREFIX = "OTROS"
AGGREGATE_CONTAINS = "PROYECTOS DE INFORMACION"
AGGREGATE_EXACT_NAMES = frozenset({"PROYECTOS FURE", "TRANQUES"})


def _aggregate_name(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    name = str(value)
    return (
        name.startswith(AGGREGATE_PREFIX)
        or AGGREGATE_CONTAINS in name
        or name in AGGREGATE_EXACT_NAMES
    )


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


def _sector_group(value: object) -> str:
    if value in {"GRAN_MINERIA_CU", "ESTATAL_CU", "MEDIANA_MINERIA_CU"}:
        return "COBRE"
    if str(value).upper() == "ORO":
        return "ORO"
    return "OTROS"


def recalculate_ols(transitions: pd.DataFrame) -> dict[str, float | int | None]:
    """Rebuild the OLS design matrix and solve it directly with least squares."""
    rows: list[dict[str, object]] = []
    for row in transitions.itertuples(index=False):
        values = (
            row.project_id,
            row.tasa_deriva,
            row.condicion_prev,
            row.capex_prev_real,
            row.pem_prev,
            row.ed_prev,
            row.delta_pem,
        )
        if any(pd.isna(value) for value in values) or float(row.capex_prev_real) <= 0:
            continue
        rows.append(
            {
                "y": float(row.tasa_deriva),
                "condicion": str(row.condicion_prev),
                "sector": _sector_group(row.sector),
                "log_capex": math.log(float(row.capex_prev_real)),
                "horizonte": float(row.pem_prev) - float(row.ed_prev),
            }
        )
    if not rows:
        return {
            "ols_coef_potencial": None,
            "ols_coef_log_capex": None,
            "ols_coef_horizonte": None,
            "ols_n": 0,
        }
    conditions = sorted({str(row["condicion"]) for row in rows} - {"BASE"})
    sectors = sorted({str(row["sector"]) for row in rows} - {"COBRE"})
    matrix_rows: list[list[float]] = []
    outcome: list[float] = []
    for row in rows:
        matrix_rows.append(
            [1.0]
            + [float(row["condicion"] == value) for value in conditions]
            + [float(row["sector"] == value) for value in sectors]
            + [float(row["log_capex"]), float(row["horizonte"])]
        )
        outcome.append(float(row["y"]))
    coefficients, _, _, _ = np.linalg.lstsq(
        np.asarray(matrix_rows, dtype=float),
        np.asarray(outcome, dtype=float),
        rcond=None,
    )
    potential_index = (
        1 + conditions.index("POTENCIAL") if "POTENCIAL" in conditions else None
    )
    return {
        "ols_coef_potencial": (
            None if potential_index is None else float(coefficients[potential_index])
        ),
        "ols_coef_log_capex": float(coefficients[-2]),
        "ols_coef_horizonte": float(coefficients[-1]),
        "ols_n": len(rows),
    }


def _cif_at_six(
    endpoints: list[dict[str, object]],
) -> tuple[float | None, float | None]:
    if not endpoints:
        return None, None
    survival = 1.0
    cif_operation = 0.0
    cif_exit = 0.0
    for time in sorted({int(row["t"]) for row in endpoints if int(row["t"]) <= 6}):
        at_risk = sum(int(row["t"]) >= time for row in endpoints)
        events_operation = sum(
            int(row["t"]) == time and int(row["evento"]) == 1
            for row in endpoints
        )
        events_exit = sum(
            int(row["t"]) == time and int(row["evento"]) == 2
            for row in endpoints
        )
        prior_survival = survival
        cif_operation += prior_survival * events_operation / at_risk
        cif_exit += prior_survival * events_exit / at_risk
        survival *= 1 - (events_operation + events_exit) / at_risk
    return cif_operation, cif_exit


def _recalculate(
    panel_path: Path,
    projects_path: Path,
    cpi_path: Path,
    egresos_path: Path | None = None,
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
    comparable_capex = 0
    unchanged_capex = 0
    updated_nominal_changes: list[float] = []
    updated_real_changes: list[float] = []
    transition_rows: list[dict[str, object]] = []
    drift_by_project: dict[object, tuple[int, float | None]] = {}
    project_histories: dict[object, tuple[pd.Series, pd.Series]] = {}
    transition_count = 0
    for project_id, group in panel.groupby("project_id", sort=False):
        rows = [row for _, row in group.sort_values("edicion", kind="stable").iterrows()]
        first = rows[0]
        last = rows[-1]
        project_histories[project_id] = (first, last)
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
                transition_delta: float | None = delta
                transition_rate: float | None = delta / (
                    int(following["edicion"]) - int(previous["edicion"])
                )
            else:
                transition_delta = None
                transition_rate = None
            transition_rows.append(
                {
                    "project_id": project_id,
                    "ed_prev": int(previous["edicion"]),
                    "pem_prev": previous["puesta_en_marcha"],
                    "delta_pem": transition_delta,
                    "tasa_deriva": transition_rate,
                    "capex_prev_real": previous["capex_real_2025"],
                    "condicion_prev": previous["condicion"],
                    "sector": previous["sector"],
                }
            )
            prior_nominal = previous["inversion_mmusd"]
            next_nominal = following["inversion_mmusd"]
            prior_capex = previous["capex_real_2025"]
            next_capex = following["capex_real_2025"]
            if pd.notna(prior_nominal) and pd.notna(next_nominal):
                comparable_capex += 1
                if float(prior_nominal) == float(next_nominal):
                    unchanged_capex += 1
                else:
                    if float(prior_nominal) != 0:
                        updated_nominal_changes.append(
                            (float(next_nominal) / float(prior_nominal) - 1) * 100
                        )
                    if (
                        pd.notna(prior_capex)
                        and pd.notna(next_capex)
                        and float(prior_capex) != 0
                    ):
                        updated_real_changes.append(
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

    next_edition = {
        edition: DATA_EDITIONS[index + 1]
        for index, edition in enumerate(DATA_EDITIONS[:-1])
    }
    project_reference = projects.drop_duplicates("project_id").set_index("project_id")
    endpoints: list[dict[str, object]] = []
    for project_id, (first, last) in project_histories.items():
        outcome = str(project_reference.loc[project_id, "desenlace"])
        first_edition = int(first["edicion"])
        last_edition = int(last["edicion"])
        if outcome in {"vigente", "censurado"}:
            duration = (2025 if outcome == "vigente" else last_edition) - first_edition
            event = 0
        else:
            duration = next_edition[last_edition] - first_edition
            if outcome == "entro_en_operacion":
                event = 1
            elif outcome == "salio_sin_operar":
                event = 2
            else:
                raise ValueError(f"desenlace desconocido para {project_id}: {outcome}")
        if duration == 0:
            continue
        condition = first["condicion"]
        if condition in {"BASE", "PROBABLE"}:
            condition_group = "BASE_PROBABLE"
        elif condition in {"POSIBLE", "POTENCIAL"}:
            condition_group = "POSIBLE_POTENCIAL"
        else:
            condition_group = None
        endpoints.append(
            {
                "first": first_edition,
                "t": duration,
                "evento": event,
                "grupo": condition_group,
            }
        )
    incident = [row for row in endpoints if int(row["first"]) >= 2015]
    base_probable = [row for row in incident if row["grupo"] == "BASE_PROBABLE"]
    possible_potential = [
        row for row in incident if row["grupo"] == "POSIBLE_POTENCIAL"
    ]
    base_cif = _cif_at_six(base_probable)
    possible_cif = _cif_at_six(possible_potential)
    ols_values = recalculate_ols(pd.DataFrame(transition_rows))

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
    egresos = pd.DataFrame()
    if egresos_path is not None:
        egresos = pd.read_csv(egresos_path)
    all_ids = {
        int(edition): set(group["project_id"])
        for edition, group in panel.groupby("edicion", sort=False)
    }
    for previous_edition, next_edition in itertools.pairwise(editions):
        previous = edition_rows.get(previous_edition, pd.DataFrame())
        following = edition_rows.get(next_edition, pd.DataFrame())
        previous_ids = set(previous.index)
        for project_id in previous_ids - all_ids.get(next_edition, set()):
            if egresos.empty:
                raise ValueError("se requiere egresos_declarados para verificar salidas")
            raw_name = previous.loc[project_id, "proyecto_raw"]
            matches = egresos.loc[
                egresos["edicion_desde"].eq(previous_edition)
                & egresos["edicion_hasta"].eq(next_edition)
                & egresos["proyecto_raw"].eq(raw_name)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"egreso no único para {previous_edition} / {raw_name}: "
                    f"{len(matches)}"
                )
            reason = str(matches.iloc[0]["motivo"])
            exits_operation += int(reason == "operacion")
            exits_without_operation += int(reason == "reestructuracion")
    result: dict[str, float | int | None] = {
        "n_proyectos": int(panel["project_id"].nunique()),
        "n_filas_agregadas_excluidas": excluded,
        "n_transiciones": transition_count,
        "media_delta_pem": _mean(deltas),
        "mediana_delta_pem": _median(deltas),
        "pct_transiciones_atraso": _percent(sum(value > 0 for value in deltas), len(deltas)),
        "pct_transiciones_adelanto": _percent(sum(value < 0 for value in deltas), len(deltas)),
        "media_tasa_deriva": _mean(rates),
        "pct_capex_sin_actualizar": _percent(unchanged_capex, comparable_capex),
        "mediana_delta_capex_nom_pct_actualizados": _median(
            updated_nominal_changes
        ),
        "mediana_delta_capex_real_pct_actualizados": _median(updated_real_changes),
        "pct_fiabilidad_a_tiempo": _percent(
            sum(value <= 0 for value in operated_drifts), len(operated_drifts)
        ),
        "pct_fiabilidad_1_anio": _percent(
            sum(value <= 1 for value in operated_drifts), len(operated_drifts)
        ),
        "mediana_deriva_acumulada_operados": _median(operated_drifts),
        "n_salidas_umbral90_sin_operar": exits_without_operation,
        "n_salidas_umbral90_operacion": exits_operation,
        "n_cohorte_incidente": len(incident),
        "cif_op_6_base_probable": base_cif[0],
        "cif_sal_6_base_probable": base_cif[1],
        "cif_op_6_posible_potencial": possible_cif[0],
        "cif_sal_6_posible_potencial": possible_cif[1],
        **ols_values,
    }
    if egresos_path is not None:
        outcome_counts = {
            "entro_en_operacion": 0,
            "salio_sin_operar": 0,
            "censurado": 0,
        }
        outcome_by_reason = {
            "operacion": "entro_en_operacion",
            "reestructuracion": "salio_sin_operar",
            "metodologia": "censurado",
            "fusion": "censurado",
            "no_declarado": "censurado",
        }
        for _, history in panel.groupby("project_id", sort=False):
            last = history.sort_values("edicion", kind="stable").iloc[-1]
            last_edition = int(last["edicion"])
            if last_edition == 2025:
                continue
            matches = egresos.loc[
                egresos["edicion_desde"].eq(last_edition)
                & egresos["proyecto_raw"].eq(last["proyecto_raw"])
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"egreso no único para {last_edition} / "
                    f"{last['proyecto_raw']}: {len(matches)}"
                )
            reason = str(matches.iloc[0]["motivo"])
            if reason not in outcome_by_reason:
                raise ValueError(f"motivo de salida desconocido: {reason}")
            outcome_counts[outcome_by_reason[reason]] += 1
        result.update(
            {
                "n_desenlace_operacion": outcome_counts["entro_en_operacion"],
                "n_desenlace_sin_operar": outcome_counts["salio_sin_operar"],
                "n_desenlace_censurado": outcome_counts["censurado"],
                "pct_salidas_confirmadas_por_nombre": (
                    None
                    if egresos.empty
                    else float(egresos["fuente"].eq("nombre").mean() * 100)
                ),
                "n_pares_conciliados": int(
                    max(panel["edicion"].nunique() - 1, 0)
                ),
            }
        )
    return result


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
    egresos_path: Path | None = None,
) -> list[dict[str, object]]:
    """Recalculate and label every claim without reading derived indicators."""
    expected = _recalculate(
        panel_path,
        projects_path,
        cpi_path,
        egresos_path,
    )
    verified: list[dict[str, object]] = []
    for claim in claims:
        item = dict(claim)
        if item.get("estado") == "informativa":
            verified.append(item)
            continue
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
