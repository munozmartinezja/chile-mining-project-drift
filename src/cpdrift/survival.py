"""Discrete-time competing-risk estimates for project outcomes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_EDITIONS = [*range(2014, 2022), 2023, 2024, 2025]
HORIZONS = (2, 4, 6)
BOOTSTRAP_REPLICATES = 1_000
BOOTSTRAP_SEED = 20261021
BOOTSTRAP_GROUPS = (
    ("incidente", "TOTAL"),
    ("incidente", "BASE_PROBABLE"),
    ("incidente", "POSIBLE_POTENCIAL"),
    ("total", "TOTAL"),
    ("total", "BASE_PROBABLE"),
    ("total", "POSIBLE_POTENCIAL"),
)
CIF_COLUMNS = [
    "cohorte",
    "grupo",
    "t",
    "en_riesgo",
    "eventos_op",
    "eventos_sal",
    "cif_op",
    "cif_sal",
    "ic_op_inf",
    "ic_op_sup",
    "ic_sal_inf",
    "ic_sal_sup",
]


def _condition_group(value: object) -> str | None:
    if value in {"BASE", "PROBABLE"}:
        return "BASE_PROBABLE"
    if value in {"POSIBLE", "POTENCIAL"}:
        return "POSIBLE_POTENCIAL"
    return None


def prepare_survival_endpoints(projects: pd.DataFrame) -> pd.DataFrame:
    """Translate project histories into one duration and outcome per project."""
    next_edition = {
        edition: DATA_EDITIONS[index + 1]
        for index, edition in enumerate(DATA_EDITIONS[:-1])
    }
    rows: list[dict[str, object]] = []
    for project in projects.itertuples(index=False):
        first = int(project.primera_edicion)
        last = int(project.ultima_edicion)
        outcome = str(project.desenlace)
        if outcome in {"vigente", "censurado"}:
            duration = (2025 if outcome == "vigente" else last) - first
            event = 0
        else:
            if last not in next_edition:
                raise ValueError(
                    f"no hay edición siguiente para {project.project_id}: {last}"
                )
            duration = next_edition[last] - first
            if outcome == "entro_en_operacion":
                event = 1
            elif outcome == "salio_sin_operar":
                event = 2
            else:
                raise ValueError(
                    f"desenlace desconocido para {project.project_id}: {outcome}"
                )
        if duration == 0:
            continue
        rows.append(
            {
                "project_id": project.project_id,
                "primera_edicion": first,
                "t": duration,
                "evento": event,
                "grupo_condicion": _condition_group(project.condicion_inicial),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "project_id",
            "primera_edicion",
            "t",
            "evento",
            "grupo_condicion",
        ],
    )


def estimate_cif(
    endpoints: pd.DataFrame,
    evaluation_times: tuple[int, ...] | list[int] | None = None,
) -> pd.DataFrame:
    """Compute the Aalen–Johansen recurrence in discrete time."""
    observed = {int(value) for value in endpoints["t"].dropna().unique()}
    times = sorted(observed | set(evaluation_times or ()))
    survival = 1.0
    cif_operation = 0.0
    cif_exit = 0.0
    rows: list[dict[str, float | int]] = []
    for time in times:
        at_risk = int(endpoints["t"].ge(time).sum())
        events_operation = int(
            (endpoints["t"].eq(time) & endpoints["evento"].eq(1)).sum()
        )
        events_exit = int(
            (endpoints["t"].eq(time) & endpoints["evento"].eq(2)).sum()
        )
        if at_risk:
            prior_survival = survival
            cif_operation += prior_survival * events_operation / at_risk
            cif_exit += prior_survival * events_exit / at_risk
            survival *= 1 - (events_operation + events_exit) / at_risk
        rows.append(
            {
                "t": time,
                "en_riesgo": at_risk,
                "eventos_op": events_operation,
                "eventos_sal": events_exit,
                "cif_op": cif_operation,
                "cif_sal": cif_exit,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_intervals(
    endpoints: pd.DataFrame,
    replicates: int,
    rng: np.random.Generator,
) -> dict[int, tuple[float, float, float, float]]:
    if endpoints.empty or replicates <= 0:
        return {}
    operation: dict[int, list[float]] = {time: [] for time in HORIZONS}
    exits: dict[int, list[float]] = {time: [] for time in HORIZONS}
    for _ in range(replicates):
        positions = rng.integers(0, len(endpoints), size=len(endpoints))
        sample = endpoints.iloc[positions].reset_index(drop=True)
        curve = estimate_cif(sample, HORIZONS).set_index("t")
        for time in HORIZONS:
            operation[time].append(float(curve.loc[time, "cif_op"]))
            exits[time].append(float(curve.loc[time, "cif_sal"]))
    return {
        time: (
            float(np.percentile(operation[time], 2.5)),
            float(np.percentile(operation[time], 97.5)),
            float(np.percentile(exits[time], 2.5)),
            float(np.percentile(exits[time], 97.5)),
        )
        for time in HORIZONS
    }


def build_cif_table(
    projects: pd.DataFrame,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Estimate CIF curves with one fixed random stream per cohort and group.

    Groups follow ``BOOTSTRAP_GROUPS``. Stream ``k`` uses a seed sequence
    ``[seed, k]``; endpoints are ordered by project ID before resampling.
    """
    endpoints = prepare_survival_endpoints(projects).sort_values(
        "project_id", kind="stable"
    ).reset_index(drop=True)
    cohort_frames = {
        "incidente": endpoints.loc[endpoints["primera_edicion"].ge(2015)],
        "total": endpoints,
    }
    outputs: list[pd.DataFrame] = []
    for stream_index, (cohort_name, group_name) in enumerate(BOOTSTRAP_GROUPS):
        cohort = cohort_frames[cohort_name]
        group = (
            cohort
            if group_name == "TOTAL"
            else cohort.loc[cohort["grupo_condicion"].eq(group_name)]
        )
        if group.empty:
            continue
        curve = estimate_cif(group, HORIZONS)
        curve.insert(0, "grupo", group_name)
        curve.insert(0, "cohorte", cohort_name)
        intervals = _bootstrap_intervals(
            group,
            bootstrap_replicates,
            np.random.default_rng([seed, stream_index]),
        )
        for column in ("ic_op_inf", "ic_op_sup", "ic_sal_inf", "ic_sal_sup"):
            curve[column] = np.nan
        for time, values in intervals.items():
            mask = curve["t"].eq(time)
            curve.loc[mask, "ic_op_inf"] = values[0]
            curve.loc[mask, "ic_op_sup"] = values[1]
            curve.loc[mask, "ic_sal_inf"] = values[2]
            curve.loc[mask, "ic_sal_sup"] = values[3]
        outputs.append(curve)
    if not outputs:
        return pd.DataFrame(columns=CIF_COLUMNS)
    return pd.concat(outputs, ignore_index=True)[CIF_COLUMNS]


def build_cif_from_parquet(
    path: Path,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Read project drift output and estimate both cohorts."""
    return build_cif_table(
        pd.read_parquet(path),
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )


def write_cif(table: pd.DataFrame, interim_dir: Path) -> Path:
    """Write the competing-risk estimates."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    path = interim_dir / "cif.parquet"
    table.to_parquet(path, index=False)
    return path
