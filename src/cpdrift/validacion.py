"""Out-of-time validation for associative delay classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrices

from .regression import EXTENDED_TERMS, fit_firth


@dataclass(frozen=True)
class ValidationTables:
    """Temporal validation metrics, calibration, and row predictions."""

    metricas: pd.DataFrame
    calibracion: pd.DataFrame
    predicciones: pd.DataFrame


@dataclass(frozen=True)
class SuccessiveValidationTables:
    """Successive-cut metrics, aggregated results, calibration, and predictions."""

    cortes: pd.DataFrame
    agregada: pd.DataFrame
    calibracion: pd.DataFrame
    predicciones: pd.DataFrame
    bootstrap: dict[str, float]


CUTS = (2018, 2019, 2020, 2021, 2023, 2024)
VALIDATION_BOOTSTRAP_SEED = 20261022
VALIDATION_BOOTSTRAP_REPLICATES = 1_000


def auc_mann_whitney(y_true: list[int] | pd.Series, scores: list[float] | pd.Series) -> float:
    """Calculate AUC using average ranks for tied scores."""
    frame = pd.DataFrame({"y": y_true, "score": scores})
    positives = int(frame["y"].eq(1).sum())
    negatives = int(frame["y"].eq(0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("AUC requiere ambas clases")
    ranks = frame["score"].rank(method="average")
    rank_sum = float(ranks[frame["y"].eq(1)].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _formula(terms: str) -> str:
    return f"atraso ~ {terms}"


def _firth_predictions(
    train: pd.DataFrame, test: pd.DataFrame, terms: str
) -> np.ndarray:
    outcome, design = dmatrices(_formula(terms), train, return_type="dataframe")
    test_design = build_design_matrices(
        [design.design_info], test, return_type="dataframe"
    )[0]
    fitted = fit_firth(design.to_numpy(), outcome.iloc[:, 0].to_numpy())
    eta = np.clip(test_design.to_numpy() @ fitted.coef, -35, 35)
    return 1 / (1 + np.exp(-eta))


def build_temporal_validation(data: pd.DataFrame) -> ValidationTables:
    """Fit pre-2020 models and evaluate them on transitions since 2020."""
    train = data.loc[data["ed_prev"].le(2019)].copy()
    test = data.loc[data["ed_prev"].ge(2020)].copy()
    if train.empty or test.empty:
        raise ValueError("validación temporal requiere entrenamiento y prueba")
    specifications = {
        "completo": EXTENDED_TERMS,
        "sin_precio": EXTENDED_TERMS.replace(" + log_precio_cu", ""),
        "solo_condicion": 'C(condicion_prev, Treatment("BASE"))',
    }
    predictions = test[["project_id", "ed_prev", "atraso"]].reset_index(drop=True)
    for model, terms in specifications.items():
        predictions[f"p_{model}"] = _firth_predictions(train, test, terms)
    training_prevalence = float(train["atraso"].mean())
    predictions["p_prevalencia"] = training_prevalence
    observed = predictions["atraso"].astype(float)
    rows = []
    for model in (*specifications, "prevalencia"):
        probability = predictions[f"p_{model}"].astype(float)
        rows.append(
            {
                "modelo": model,
                "auc": auc_mann_whitney(observed, probability),
                "brier": float(((observed - probability) ** 2).mean()),
                "calibracion_global": float(probability.mean() - observed.mean()),
                "n_train": len(train),
                "n_test": len(test),
                "prevalencia_train": training_prevalence,
                "prevalencia_test": float(observed.mean()),
            }
        )
    calibration_source = predictions.copy()
    calibration_source["quintil"] = pd.qcut(
        predictions["p_completo"].rank(method="first"), 5, labels=False
    ) + 1
    calibration = (
        calibration_source.groupby("quintil", as_index=False)
        .agg(
            media_predicha=("p_completo", "mean"),
            tasa_observada=("atraso", "mean"),
            n=("atraso", "size"),
        )
        .astype({"quintil": int, "n": int})
    )
    return ValidationTables(pd.DataFrame(rows), calibration, predictions)


def _metrics(
    predictions: pd.DataFrame,
    model: str,
    n_train: int | None = None,
) -> dict[str, float | int | str]:
    observed = predictions["atraso"].astype(float)
    probability = predictions[f"p_{model}"].astype(float)
    return {
        "modelo": model,
        "n_train": n_train if n_train is not None else np.nan,
        "n_test": len(predictions),
        "prevalencia_test": float(observed.mean()),
        "auc": np.nan if model == "prevalencia" else auc_mann_whitney(observed, probability),
        "brier": float(((observed - probability) ** 2).mean()),
        "calibracion": float(probability.mean() - observed.mean()),
    }


def resample_project_predictions(
    predictions: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Sample sorted project clusters and retain every row in each sampled cluster."""
    identifiers = np.asarray(sorted(predictions["project_id"].unique(), key=str))
    sampled = rng.choice(identifiers, size=len(identifiers), replace=True)
    pieces = []
    for draw, project_id in enumerate(sampled):
        piece = predictions.loc[predictions["project_id"].eq(project_id)].copy()
        piece["bootstrap_draw"] = draw
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def bootstrap_aggregated_predictions(
    predictions: pd.DataFrame,
    replicates: int = VALIDATION_BOOTSTRAP_REPLICATES,
    seed: int = VALIDATION_BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Return project-bootstrap intervals for aggregate discrimination and Brier gain."""
    rng = np.random.default_rng(seed)
    auc_values: list[float] = []
    improvement_values: list[float] = []
    for _ in range(replicates):
        sample = resample_project_predictions(predictions, rng)
        if sample["atraso"].nunique() < 2:
            continue
        observed = sample["atraso"].astype(float)
        auc_values.append(auc_mann_whitney(observed, sample["p_completo"]))
        complete = float(((observed - sample["p_completo"]) ** 2).mean())
        prevalence = float(((observed - sample["p_prevalencia"]) ** 2).mean())
        improvement_values.append(prevalence - complete)
    if not auc_values:
        raise ValueError("bootstrap agregado sin réplicas con ambas clases")
    return {
        "auc_ic_inf": float(np.percentile(auc_values, 2.5)),
        "auc_ic_sup": float(np.percentile(auc_values, 97.5)),
        "mejora_brier_ic_inf": float(np.percentile(improvement_values, 2.5)),
        "mejora_brier_ic_sup": float(np.percentile(improvement_values, 97.5)),
    }


def build_successive_validation(
    data: pd.DataFrame,
    bootstrap_replicates: int = VALIDATION_BOOTSTRAP_REPLICATES,
    seed: int = VALIDATION_BOOTSTRAP_SEED,
) -> SuccessiveValidationTables:
    """Evaluate expanding-window models on six successive edition cuts."""
    specifications = {
        "completo": EXTENDED_TERMS,
        "sin_precio": EXTENDED_TERMS.replace(" + log_precio_cu", ""),
        "solo_condicion": 'C(condicion_prev, Treatment("BASE"))',
    }
    cut_metrics = []
    prediction_frames = []
    for cut in CUTS:
        train = data.loc[data["ed_prev"].lt(cut)].copy()
        test = data.loc[data["ed_prev"].eq(cut)].copy()
        if train.empty or test.empty:
            raise ValueError(f"corte {cut} requiere entrenamiento y prueba")
        predictions = test[["project_id", "ed_prev", "atraso"]].reset_index(drop=True)
        predictions.insert(0, "corte", cut)
        for model, terms in specifications.items():
            predictions[f"p_{model}"] = _firth_predictions(train, test, terms)
        predictions["p_prevalencia"] = float(train["atraso"].mean())
        for model in (*specifications, "prevalencia"):
            row = _metrics(predictions, model, len(train))
            row["corte"] = cut
            cut_metrics.append(row)
        prediction_frames.append(predictions)
    combined = pd.concat(prediction_frames, ignore_index=True)
    aggregate = pd.DataFrame(
        [_metrics(combined, model) for model in (*specifications, "prevalencia")]
    )
    calibration_source = combined.copy()
    calibration_source["quintil"] = (
        pd.qcut(
            calibration_source["p_completo"].rank(method="first"),
            5,
            labels=False,
        )
        + 1
    )
    calibration = (
        calibration_source.groupby("quintil", as_index=False)
        .agg(
            media_predicha=("p_completo", "mean"),
            tasa_observada=("atraso", "mean"),
            n=("atraso", "size"),
        )
        .astype({"quintil": int, "n": int})
    )
    bootstrap = bootstrap_aggregated_predictions(combined, bootstrap_replicates, seed)
    return SuccessiveValidationTables(
        pd.DataFrame(cut_metrics), aggregate, calibration, combined, bootstrap
    )


def write_temporal_validation(tables: ValidationTables, interim_dir: Path) -> None:
    """Write all temporal validation outputs."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    tables.metricas.to_parquet(interim_dir / "validacion_temporal.parquet", index=False)
    tables.calibracion.to_parquet(interim_dir / "calibracion_test.parquet", index=False)
    tables.predicciones.to_parquet(interim_dir / "predicciones_test.parquet", index=False)


def write_successive_validation(
    tables: SuccessiveValidationTables, interim_dir: Path
) -> None:
    """Write successive-cut validation and its aggregate views."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    tables.cortes.to_parquet(interim_dir / "validacion_cortes.parquet", index=False)
    tables.agregada.to_parquet(interim_dir / "validacion_agregada.parquet", index=False)
    tables.calibracion.to_parquet(
        interim_dir / "calibracion_agregada.parquet", index=False
    )
    tables.predicciones.to_parquet(
        interim_dir / "predicciones_cortes.parquet", index=False
    )
