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


def write_temporal_validation(tables: ValidationTables, interim_dir: Path) -> None:
    """Write all temporal validation outputs."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    tables.metricas.to_parquet(interim_dir / "validacion_temporal.parquet", index=False)
    tables.calibracion.to_parquet(interim_dir / "calibracion_test.parquet", index=False)
    tables.predicciones.to_parquet(interim_dir / "predicciones_test.parquet", index=False)
