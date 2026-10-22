"""Independent recomputation of headline figures from primary output tables."""

from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrices

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
    "n_train_corte_unico",
    "n_test_corte_unico",
    "n_cortes",
    "n_test_agregado",
    "n_ediciones",
    "min_proyectos_edicion",
    "max_proyectos_edicion",
    "n_escenarios_regresion",
    "n_escenarios_potencial_robusto",
    "n_proyectos_anio4",
    "n_cifras_verificadas",
}

DATA_EDITIONS = [*range(2014, 2022), 2023, 2024, 2025]
AGGREGATE_PREFIX = "OTROS"
AGGREGATE_CONTAINS = "PROYECTOS DE INFORMACION"
AGGREGATE_EXACT_NAMES = frozenset({"PROYECTOS FURE", "TRANQUES"})
VERIFY_TERMS = (
    'C(condicion_prev, Treatment("BASE"))'
    ' + C(grupo_sector, Treatment("COBRE"))'
    " + log_capex + horizonte + estatal + log_precio_cu"
)


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


def _cif_at(
    endpoints: list[dict[str, object]], horizon: int
) -> tuple[float | None, float | None]:
    if not endpoints:
        return None, None
    survival = 1.0
    operation = 0.0
    exits = 0.0
    for time in sorted({int(row["t"]) for row in endpoints if int(row["t"]) <= horizon}):
        at_risk = sum(int(row["t"]) >= time for row in endpoints)
        d_operation = sum(
            int(row["t"]) == time and int(row["evento"]) == 1 for row in endpoints
        )
        d_exit = sum(
            int(row["t"]) == time and int(row["evento"]) == 2 for row in endpoints
        )
        operation += survival * d_operation / at_risk
        exits += survival * d_exit / at_risk
        survival *= 1 - (d_operation + d_exit) / at_risk
    return operation, exits


def _bootstrap_cif(
    endpoints: list[dict[str, object]], stream_index: int
) -> dict[str, float]:
    ordered = sorted(endpoints, key=lambda row: str(row["project_id"]))
    rng = np.random.default_rng([20261021, stream_index])
    values = {key: [] for key in ("op_4", "sal_4", "op_6", "sal_6")}
    for _ in range(1_000):
        positions = rng.integers(0, len(ordered), size=len(ordered))
        sample = [ordered[int(position)] for position in positions]
        for horizon in (4, 6):
            operation, exits = _cif_at(sample, horizon)
            values[f"op_{horizon}"].append(float(operation))
            values[f"sal_{horizon}"].append(float(exits))
    output = {}
    for event in ("op", "sal"):
        for horizon in (4, 6):
            sample_values = values[f"{event}_{horizon}"]
            output[f"ic_{event}_inf_{horizon}"] = float(
                np.percentile(sample_values, 2.5)
            )
            output[f"ic_{event}_sup_{horizon}"] = float(
                np.percentile(sample_values, 97.5)
            )
    return output


def _verify_prices(
    workbook_path: Path, cpi_path: Path, editions: set[int]
) -> pd.DataFrame:
    sheet = pd.read_excel(workbook_path, sheet_name="Monthly Prices", header=None)
    copper_columns = []
    for row in range(min(8, len(sheet) - 1)):
        for column in range(sheet.shape[1]):
            if (
                str(sheet.iat[row, column]).strip() == "Copper"
                and str(sheet.iat[row + 1, column]).strip() == "($/mt)"
            ):
                copper_columns.append(column)
    if len(copper_columns) != 1:
        raise ValueError("columna Copper ($/mt) no única")
    pattern = re.compile(r"^(\d{4})M(\d{2})$")
    period_counts = {
        column: sum(
            bool(pattern.fullmatch(str(value).strip()))
            for value in sheet.iloc[:, column]
            if pd.notna(value)
        )
        for column in range(sheet.shape[1])
    }
    period_column = max(period_counts, key=period_counts.get)
    monthly = {}
    for period, value in zip(
        sheet.iloc[:, period_column], sheet.iloc[:, copper_columns[0]], strict=True
    ):
        match = pattern.fullmatch(str(period).strip())
        if match and pd.notna(value):
            monthly[(int(match.group(1)), int(match.group(2)))] = float(value)
    cpi_frame = pd.read_csv(cpi_path)
    cpi_s1 = {int(row.anio): float(row.s1) for row in cpi_frame.itertuples()}
    row_2025 = cpi_frame.loc[cpi_frame["anio"].eq(2025)].iloc[0]
    annual_2025 = float((row_2025["s1"] + row_2025["s2"]) / 2)
    rows = []
    for edition in sorted(editions):
        values = [monthly[(edition, month)] for month in range(1, 7)]
        rows.append(
            {
                "ed_prev": edition,
                "log_precio_cu": math.log(
                    sum(values) / 6 * annual_2025 / cpi_s1[edition]
                ),
            }
        )
    return pd.DataFrame(rows)


def _verify_extended_data(
    transitions: pd.DataFrame,
    panel: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    data = transitions.copy()
    data["grupo_sector"] = data["sector"].map(_sector_group)
    data["log_capex"] = pd.to_numeric(
        data["capex_prev_real"], errors="coerce"
    ).map(lambda value: math.log(value) if pd.notna(value) and value > 0 else math.nan)
    data["horizonte"] = pd.to_numeric(data["pem_prev"], errors="coerce") - pd.to_numeric(
        data["ed_prev"], errors="coerce"
    )
    data["atraso"] = pd.to_numeric(data["delta_pem"], errors="coerce").gt(0).astype(int)
    operators = panel[["project_id", "edicion", "operador_norm"]].drop_duplicates(
        ["project_id", "edicion"]
    ).rename(columns={"edicion": "ed_prev"})
    data = data.merge(operators, on=["project_id", "ed_prev"], validate="many_to_one")
    operator = data["operador_norm"].fillna("").astype(str).str.upper()
    data["estatal"] = operator.str.contains(r"CODELCO|ENAMI", regex=True).astype(int)
    data = data.merge(prices, on="ed_prev", validate="many_to_one")
    return data.dropna(
        subset=[
            "project_id",
            "tasa_deriva",
            "condicion_prev",
            "grupo_sector",
            "log_capex",
            "horizonte",
            "log_precio_cu",
        ]
    )


def _clustered_ols(
    data: pd.DataFrame, formula: str
) -> tuple[pd.Series, pd.Series]:
    outcome, design = dmatrices(formula, data, return_type="dataframe")
    x = design.to_numpy()
    y = outcome.iloc[:, 0].to_numpy()
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    bread = np.linalg.pinv(x.T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]))
    groups = data.loc[design.index, "project_id"]
    for group in groups.unique():
        mask = groups.eq(group).to_numpy()
        score = x[mask].T @ residual[mask]
        meat += np.outer(score, score)
    group_count = groups.nunique()
    correction = group_count / (group_count - 1) * (len(x) - 1) / (len(x) - x.shape[1])
    covariance = correction * bread @ meat @ bread
    se = np.sqrt(np.clip(np.diag(covariance), 0, None))
    p = np.array([math.erfc(abs(value) / math.sqrt(2)) for value in beta / se])
    return pd.Series(beta, index=design.columns), pd.Series(p, index=design.columns)


def _verify_firth(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError("Firth independiente requiere rango completo")
    beta = np.zeros(x.shape[1])
    def objective(candidate: np.ndarray) -> float:
        eta = np.clip(x @ candidate, -35, 35)
        probability = 1 / (1 + np.exp(-eta))
        weights = np.clip(probability * (1 - probability), 1e-12, None)
        sign, logdet = np.linalg.slogdet(x.T @ (weights[:, None] * x))
        return (
            -np.inf
            if sign <= 0
            else float(np.sum(y * eta - np.logaddexp(0, eta)) + 0.5 * logdet)
        )

    current = objective(beta)
    for _ in range(100):
        probability = 1 / (1 + np.exp(-np.clip(x @ beta, -35, 35)))
        weights = np.clip(probability * (1 - probability), 1e-12, None)
        information = x.T @ (weights[:, None] * x)
        inverse = np.linalg.pinv(information)
        leverage = np.sum((x @ inverse) * x, axis=1) * weights
        step = inverse @ (x.T @ (y - probability + leverage * (0.5 - probability)))
        scale = 1.0
        candidate = beta + step
        candidate_value = objective(candidate)
        while candidate_value < current - 1e-12 and scale > 2**-30:
            scale /= 2
            candidate = beta + scale * step
            candidate_value = objective(candidate)
        beta = candidate
        current = candidate_value
        if np.max(np.abs(scale * step)) < 1e-8:
            return beta
    raise ValueError("Firth independiente no converge")


def _verify_auc(y: pd.Series, probability: np.ndarray) -> float:
    ranks = pd.Series(probability).rank(method="average")
    positive = y.reset_index(drop=True).eq(1)
    n_positive = int(positive.sum())
    n_negative = len(y) - n_positive
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def _recalculate(
    panel_path: Path,
    projects_path: Path,
    cpi_path: Path,
    egresos_path: Path | None = None,
    copper_workbook_path: Path | None = None,
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
                "project_id": project_id,
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
    base_cif_four = _cif_at(base_probable, 4)
    possible_cif_four = _cif_at(possible_potential, 4)
    base_intervals = _bootstrap_cif(base_probable, 1) if base_probable else {}
    possible_intervals = (
        _bootstrap_cif(possible_potential, 2) if possible_potential else {}
    )
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
        "cif_op_4_base_probable": base_cif_four[0],
        "cif_sal_4_base_probable": base_cif_four[1],
        "cif_op_4_posible_potencial": possible_cif_four[0],
        "cif_sal_4_posible_potencial": possible_cif_four[1],
        **{
            f"{claim_id}_base_probable": value
            for claim_id, value in base_intervals.items()
        },
        **{
            f"{claim_id}_posible_potencial": value
            for claim_id, value in possible_intervals.items()
        },
        **ols_values,
    }
    valid_deltas = [value for value in deltas if value is not None]
    for condition in ("BASE", "PROBABLE", "POSIBLE", "POTENCIAL"):
        condition_deltas = [
            float(row["delta_pem"])
            for row in transition_rows
            if row["condicion_prev"] == condition and row["delta_pem"] is not None
        ]
        result[f"pct_atraso_{condition.lower()}"] = _percent(
            sum(value > 0 for value in condition_deltas), len(condition_deltas)
        )
    initial_capex = [
        float(first["capex_real_2025"])
        for first, _ in project_histories.values()
        if pd.notna(first["capex_real_2025"])
    ]
    copper_sectors = {"GRAN_MINERIA_CU", "ESTATAL_CU", "MEDIANA_MINERIA_CU"}
    first_rows = [first for first, _ in project_histories.values()]
    edition_counts = panel.groupby("edicion")["project_id"].nunique()
    result.update(
        {
            "pct_transiciones_sin_cambio": _percent(
                sum(value == 0 for value in valid_deltas), len(valid_deltas)
            ),
            "mediana_capex_inicial_real": _median(initial_capex),
            "pct_proyectos_cobre": _percent(
                sum(row["sector"] in copper_sectors for row in first_rows),
                len(first_rows),
            ),
            "pct_proyectos_antofagasta": _percent(
                sum(str(row["region"]).upper() == "ANTOFAGASTA" for row in first_rows),
                len(first_rows),
            ),
            "n_ediciones": int(panel["edicion"].nunique()),
            "min_proyectos_edicion": int(edition_counts.min()),
            "max_proyectos_edicion": int(edition_counts.max()),
        }
    )
    year_four_drifts = []
    for first, _last in project_histories.values():
        project_id = first["project_id"]
        first_edition = int(first["edicion"])
        first_pem = first["puesta_en_marcha"]
        appearances = panel.loc[panel["project_id"].eq(project_id)]
        target = appearances.loc[appearances["edicion"].eq(first_edition + 4)]
        if not target.empty and pd.notna(first_pem) and pd.notna(target.iloc[0]["puesta_en_marcha"]):
            year_four_drifts.append(float(target.iloc[0]["puesta_en_marcha"]) - float(first_pem))
    if len(year_four_drifts) >= 10:
        result.update(
            {
                "mediana_deriva_anio4": float(np.percentile(year_four_drifts, 50)),
                "p90_deriva_anio4": float(np.percentile(year_four_drifts, 90)),
                "n_proyectos_anio4": len(year_four_drifts),
            }
        )
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
    if copper_workbook_path is not None:
        transitions = pd.DataFrame(transition_rows)
        prices = _verify_prices(
            copper_workbook_path,
            cpi_path,
            {int(value) for value in panel["edicion"].unique()},
        )
        data = _verify_extended_data(transitions, panel, prices)
        formula = f"tasa_deriva ~ {VERIFY_TERMS}"
        coefficients, p_values = _clustered_ols(data, formula)
        potential = 'C(condicion_prev, Treatment("BASE"))[T.POTENCIAL]'
        result.update(
            {
                "ols_amp_coef_potencial": float(coefficients[potential]),
                "ols_amp_coef_precio_cu": float(coefficients["log_precio_cu"]),
                "ols_amp_p_precio_cu": float(p_values["log_precio_cu"]),
                "ols_amp_coef_estatal": float(coefficients["estatal"]),
                "ols_amp_p_estatal": float(p_values["estatal"]),
            }
        )
        train = data.loc[data["ed_prev"].le(2019)]
        test = data.loc[data["ed_prev"].ge(2020)]
        prediction_map: dict[str, np.ndarray] = {}
        for name, terms in {
            "completo": VERIFY_TERMS,
            "solo_condicion": 'C(condicion_prev, Treatment("BASE"))',
        }.items():
            y_train, x_train = dmatrices(
                f"atraso ~ {terms}", train, return_type="dataframe"
            )
            x_test = build_design_matrices(
                [x_train.design_info], test, return_type="dataframe"
            )[0]
            beta = _verify_firth(
                x_train.to_numpy(), y_train.iloc[:, 0].to_numpy()
            )
            prediction_map[name] = 1 / (
                1 + np.exp(-np.clip(x_test.to_numpy() @ beta, -35, 35))
            )
        observed = test["atraso"].astype(float).reset_index(drop=True)
        prevalence = float(train["atraso"].mean())
        complete_probability = prediction_map["completo"]
        result.update(
            {
                "auc_corte_unico_completo": _verify_auc(observed, complete_probability),
                "auc_corte_unico_solo_condicion": _verify_auc(
                    observed, prediction_map["solo_condicion"]
                ),
                "brier_corte_unico_completo": float(
                    ((observed - complete_probability) ** 2).mean()
                ),
                "brier_corte_unico_prevalencia": float(
                    ((observed - prevalence) ** 2).mean()
                ),
                "calibracion_corte_unico_completo": float(
                    complete_probability.mean() - observed.mean()
                ),
                "n_train_corte_unico": len(train),
                "n_test_corte_unico": len(test),
            }
        )
        cut_rows: list[dict[str, object]] = []
        cut_aucs: list[float] = []
        for cut in (2018, 2019, 2020, 2021, 2023, 2024):
            cut_train = data.loc[data["ed_prev"].lt(cut)]
            cut_test = data.loc[data["ed_prev"].eq(cut)]
            if cut_train.empty or cut_test.empty:
                raise ValueError(f"corte {cut} requiere entrenamiento y prueba")
            probabilities: dict[str, np.ndarray] = {}
            for name, terms in {
                "completo": VERIFY_TERMS,
                "solo_condicion": 'C(condicion_prev, Treatment("BASE"))',
            }.items():
                y_cut, x_cut = dmatrices(
                    f"atraso ~ {terms}", cut_train, return_type="dataframe"
                )
                x_future = build_design_matrices(
                    [x_cut.design_info], cut_test, return_type="dataframe"
                )[0]
                beta = _verify_firth(
                    x_cut.to_numpy(), y_cut.iloc[:, 0].to_numpy()
                )
                probabilities[name] = 1 / (
                    1 + np.exp(-np.clip(x_future.to_numpy() @ beta, -35, 35))
                )
            cut_observed = cut_test["atraso"].astype(float).reset_index(drop=True)
            cut_auc = _verify_auc(cut_observed, probabilities["completo"])
            cut_aucs.append(cut_auc)
            prevalence_probability = float(cut_train["atraso"].mean())
            for position, (_, source_row) in enumerate(cut_test.iterrows()):
                cut_rows.append(
                    {
                        "project_id": source_row["project_id"],
                        "atraso": float(source_row["atraso"]),
                        "p_completo": float(probabilities["completo"][position]),
                        "p_solo_condicion": float(
                            probabilities["solo_condicion"][position]
                        ),
                        "p_prevalencia": prevalence_probability,
                    }
                )
        combined = pd.DataFrame(cut_rows)
        aggregate_observed = combined["atraso"]
        aggregate_auc = _verify_auc(aggregate_observed, combined["p_completo"].to_numpy())
        aggregate_condition_auc = _verify_auc(
            aggregate_observed, combined["p_solo_condicion"].to_numpy()
        )
        complete_brier = float(
            ((aggregate_observed - combined["p_completo"]) ** 2).mean()
        )
        prevalence_brier = float(
            ((aggregate_observed - combined["p_prevalencia"]) ** 2).mean()
        )
        rng = np.random.default_rng(20261022)
        identifiers = sorted(combined["project_id"].unique(), key=str)
        bootstrap_auc: list[float] = []
        bootstrap_improvement: list[float] = []
        for _ in range(1_000):
            sampled_ids = rng.choice(identifiers, size=len(identifiers), replace=True)
            sample = pd.concat(
                [combined.loc[combined["project_id"].eq(item)] for item in sampled_ids],
                ignore_index=True,
            )
            if sample["atraso"].nunique() < 2:
                continue
            sampled_observed = sample["atraso"]
            bootstrap_auc.append(
                _verify_auc(sampled_observed, sample["p_completo"].to_numpy())
            )
            bootstrap_improvement.append(
                float(((sampled_observed - sample["p_prevalencia"]) ** 2).mean())
                - float(((sampled_observed - sample["p_completo"]) ** 2).mean())
            )
        result.update(
            {
                "auc_agregado_completo": aggregate_auc,
                "auc_agregado_completo_ic_inf": float(
                    np.percentile(bootstrap_auc, 2.5)
                ),
                "auc_agregado_completo_ic_sup": float(
                    np.percentile(bootstrap_auc, 97.5)
                ),
                "auc_agregado_solo_condicion": aggregate_condition_auc,
                "brier_agregado_completo": complete_brier,
                "brier_agregado_prevalencia": prevalence_brier,
                "mejora_brier_agregada": prevalence_brier - complete_brier,
                "mejora_brier_agregada_ic_inf": float(
                    np.percentile(bootstrap_improvement, 2.5)
                ),
                "mejora_brier_agregada_ic_sup": float(
                    np.percentile(bootstrap_improvement, 97.5)
                ),
                "calibracion_agregada_completo": float(
                    combined["p_completo"].mean() - aggregate_observed.mean()
                ),
                "auc_corte_min": min(cut_aucs),
                "auc_corte_max": max(cut_aucs),
                "n_cortes": 6,
                "n_test_agregado": len(combined),
            }
        )
        scenario_specs = {
            "principal": (data, formula),
            "sin_par_2021_2023": (data.loc[~data["ed_prev"].eq(2021)], formula),
            "sin_estatales": (
                data.loc[data["estatal"].eq(0)],
                formula.replace(" + estatal", ""),
            ),
        }
        robust = 0
        for sample, scenario_formula in scenario_specs.values():
            scenario_coef, scenario_p = _clustered_ols(sample, scenario_formula)
            robust += int(
                scenario_coef[potential] > 0 and scenario_p[potential] < 0.05
            )
        result["n_escenarios_regresion"] = len(scenario_specs)
        result["n_escenarios_potencial_robusto"] = robust
        if egresos_path is not None:
            comparable = egresos.loc[
                egresos["motivo"].isin({"operacion", "reestructuracion"})
            ]
            discrepancies = 0
            for exit_row in comparable.itertuples(index=False):
                source = panel.loc[
                    panel["edicion"].eq(exit_row.edicion_desde)
                    & panel["proyecto_raw"].eq(exit_row.proyecto_raw)
                ].iloc[0]
                rule = (
                    "operacion"
                    if pd.notna(source["puesta_en_marcha"])
                    and float(source["puesta_en_marcha"])
                    <= int(exit_row.edicion_hasta)
                    else "reestructuracion"
                )
                discrepancies += int(rule != exit_row.motivo)
            result["pct_regla_fecha_discrepante"] = _percent(
                discrepancies, len(comparable)
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
    copper_workbook_path: Path | None = None,
) -> list[dict[str, object]]:
    """Recalculate and label every claim without reading derived indicators."""
    expected = _recalculate(
        panel_path,
        projects_path,
        cpi_path,
        egresos_path,
        copper_workbook_path,
    )
    verified: list[dict[str, object]] = []
    count_claim: dict[str, object] | None = None
    for claim in claims:
        item = dict(claim)
        if item.get("id") == "n_cifras_verificadas":
            count_claim = item
            continue
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
    if count_claim is not None:
        recalculated = sum(item.get("estado") == "verificada" for item in verified)
        count_claim["valor_recalculado"] = recalculated
        count_claim["estado"] = (
            "verificada"
            if _matches("n_cifras_verificadas", count_claim["valor"], recalculated)
            else "fallida"
        )
        verified.append(count_claim)
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
