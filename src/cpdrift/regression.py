"""Associative regression models for schedule drift transitions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrices

FORMULA_TERMS = (
    'C(condicion_prev, Treatment("BASE"))'
    ' + C(grupo_sector, Treatment("COBRE"))'
    " + log_capex + horizonte"
)
OLS_FORMULA = f"tasa_deriva ~ {FORMULA_TERMS}"
LOGIT_FORMULA = f"atraso ~ {FORMULA_TERMS}"
EXTENDED_TERMS = FORMULA_TERMS + " + estatal + log_precio_cu"
OLS_EXTENDED_FORMULA = f"tasa_deriva ~ {EXTENDED_TERMS}"
LOGIT_EXTENDED_FORMULA = f"atraso ~ {EXTENDED_TERMS}"
RESULT_COLUMNS = [
    "termino",
    "coef",
    "ee",
    "ic_inf",
    "ic_sup",
    "p",
    "n",
    "r2",
]
LOGIT_COLUMNS = [
    "termino",
    "coef",
    "ee",
    "ic_inf",
    "ic_sup",
    "p",
    "n",
    "pseudo_r2",
    "ee_tipo",
]


@dataclass(frozen=True)
class RegressionTables:
    """Tabular OLS and logistic model results."""

    ols: pd.DataFrame
    ols_ampliado: pd.DataFrame
    logit: pd.DataFrame


@dataclass(frozen=True)
class FirthResult:
    """Numerical result of a Firth penalized logistic fit."""

    coef: np.ndarray
    se: np.ndarray
    converged: bool
    iterations: int
    penalized_loglike: float


def _sector_group(value: object) -> str:
    if value in {"GRAN_MINERIA_CU", "ESTATAL_CU", "MEDIANA_MINERIA_CU"}:
        return "COBRE"
    if str(value).upper() == "ORO":
        return "ORO"
    return "OTROS"


def prepare_regression_data(transitions: pd.DataFrame) -> pd.DataFrame:
    """Create the complete-case design variables shared by both models."""
    data = transitions.copy()
    data["grupo_sector"] = data["sector"].map(_sector_group)
    data["log_capex"] = np.log(
        pd.to_numeric(data["capex_prev_real"], errors="coerce").where(
            lambda values: values.gt(0)
        )
    )
    data["horizonte"] = pd.to_numeric(
        data["pem_prev"], errors="coerce"
    ) - pd.to_numeric(data["ed_prev"], errors="coerce")
    data["atraso"] = pd.to_numeric(data["delta_pem"], errors="coerce").gt(0).astype(int)
    required = [
        "project_id",
        "tasa_deriva",
        "condicion_prev",
        "grupo_sector",
        "log_capex",
        "horizonte",
        "delta_pem",
    ]
    return data.dropna(subset=required).copy()


def prepare_extended_regression_data(
    transitions: pd.DataFrame,
    panel: pd.DataFrame,
    copper_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Add ownership and copper-price variables to the principal design."""
    data = prepare_regression_data(transitions)
    operators = panel[["project_id", "edicion", "operador_norm"]].drop_duplicates(
        ["project_id", "edicion"]
    )
    operators = operators.rename(columns={"edicion": "ed_prev"})
    data = data.merge(
        operators,
        on=["project_id", "ed_prev"],
        how="left",
        validate="many_to_one",
    )
    data = data.merge(
        copper_prices[["edicion", "precio_real_2025"]].rename(
            columns={"edicion": "ed_prev"}
        ),
        on="ed_prev",
        how="left",
        validate="many_to_one",
    )
    operator = data["operador_norm"].fillna("").astype(str).str.upper()
    data["estatal"] = operator.str.contains(r"CODELCO|ENAMI", regex=True).astype(int)
    data["log_precio_cu"] = np.log(
        pd.to_numeric(data["precio_real_2025"], errors="coerce").where(
            lambda values: values.gt(0)
        )
    )
    return data.dropna(subset=["log_precio_cu"]).copy()


def _penalized_loglike(x: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    eta = np.clip(x @ beta, -35, 35)
    loglike = float(np.sum(y * eta - np.logaddexp(0, eta)))
    probability = 1 / (1 + np.exp(-eta))
    weights = np.clip(probability * (1 - probability), 1e-12, None)
    sign, logdet = np.linalg.slogdet(x.T @ (weights[:, None] * x))
    return -np.inf if sign <= 0 else loglike + 0.5 * float(logdet)


def fit_firth(
    x: np.ndarray,
    y: np.ndarray,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> FirthResult:
    """Fit Firth logistic regression with adjusted-score Newton updates."""
    design = np.asarray(x, dtype=float)
    outcome = np.asarray(y, dtype=float)
    if design.ndim != 2 or len(outcome) != len(design):
        raise ValueError("dimensiones inválidas para Firth")
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Firth requiere una matriz de diseño de rango completo")
    beta = np.zeros(design.shape[1], dtype=float)
    current = _penalized_loglike(design, outcome, beta)
    for iteration in range(1, max_iterations + 1):
        eta = np.clip(design @ beta, -35, 35)
        probability = 1 / (1 + np.exp(-eta))
        weights = np.clip(probability * (1 - probability), 1e-12, None)
        information = design.T @ (weights[:, None] * design)
        inverse = np.linalg.pinv(information)
        leverage = np.sum((design @ inverse) * design, axis=1) * weights
        score = design.T @ (
            outcome - probability + leverage * (0.5 - probability)
        )
        step = inverse @ score
        scale = 1.0
        candidate = beta + step
        candidate_value = _penalized_loglike(design, outcome, candidate)
        while candidate_value < current - 1e-12 and scale > 2**-30:
            scale /= 2
            candidate = beta + scale * step
            candidate_value = _penalized_loglike(design, outcome, candidate)
        if candidate_value < current - 1e-12:
            if np.max(np.abs(scale * step)) < tolerance:
                candidate = beta
                candidate_value = current
            else:
                raise ValueError("Firth no pudo mejorar la verosimilitud penalizada")
        beta = candidate
        current = candidate_value
        if np.max(np.abs(scale * step)) < tolerance:
            eta = np.clip(design @ beta, -35, 35)
            probability = 1 / (1 + np.exp(-eta))
            weights = np.clip(probability * (1 - probability), 1e-12, None)
            covariance = np.linalg.pinv(design.T @ (weights[:, None] * design))
            return FirthResult(
                beta,
                np.sqrt(np.clip(np.diag(covariance), 0, None)),
                True,
                iteration,
                current,
            )
    raise ValueError("Firth no converge en 100 iteraciones")


def fit_ols(transitions: pd.DataFrame) -> pd.DataFrame:
    """Fit the associative linear model with project-clustered uncertainty."""
    data = prepare_regression_data(transitions)
    if data.empty:
        raise ValueError("sin transiciones completas para OLS")
    fitted = smf.ols(OLS_FORMULA, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["project_id"]}
    )
    intervals = fitted.conf_int()
    return pd.DataFrame(
        {
            "termino": fitted.params.index,
            "coef": fitted.params.to_numpy(dtype=float),
            "ee": fitted.bse.to_numpy(dtype=float),
            "ic_inf": intervals.iloc[:, 0].to_numpy(dtype=float),
            "ic_sup": intervals.iloc[:, 1].to_numpy(dtype=float),
            "p": fitted.pvalues.to_numpy(dtype=float),
            "n": int(fitted.nobs),
            "r2": float(fitted.rsquared),
        },
        columns=RESULT_COLUMNS,
    )


def fit_extended_ols(data: pd.DataFrame) -> pd.DataFrame:
    """Fit the expanded associative OLS with clustered uncertainty."""
    if data.empty:
        raise ValueError("sin transiciones completas para OLS ampliado")
    fitted = smf.ols(OLS_EXTENDED_FORMULA, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["project_id"]}
    )
    intervals = fitted.conf_int()
    return pd.DataFrame(
        {
            "termino": fitted.params.index,
            "coef": fitted.params.to_numpy(dtype=float),
            "ee": fitted.bse.to_numpy(dtype=float),
            "ic_inf": intervals.iloc[:, 0].to_numpy(dtype=float),
            "ic_sup": intervals.iloc[:, 1].to_numpy(dtype=float),
            "p": fitted.pvalues.to_numpy(dtype=float),
            "n": int(fitted.nobs),
            "r2": float(fitted.rsquared),
        },
        columns=RESULT_COLUMNS,
    )


def fit_logit(data: pd.DataFrame) -> pd.DataFrame:
    """Fit the expanded Firth logit and report odds ratios."""
    if data.empty or data["atraso"].nunique() < 2:
        raise ValueError("se requieren ambos valores de atraso para logit")
    outcome, design = dmatrices(LOGIT_EXTENDED_FORMULA, data, return_type="dataframe")
    fitted = fit_firth(design.to_numpy(), outcome.iloc[:, 0].to_numpy())
    z_scores = fitted.coef / fitted.se
    p_values = np.array([math.erfc(abs(value) / math.sqrt(2)) for value in z_scores])
    ordinary_eta = np.clip(design.to_numpy() @ fitted.coef, -35, 35)
    ordinary_ll = float(
        np.sum(outcome.iloc[:, 0] * ordinary_eta - np.logaddexp(0, ordinary_eta))
    )
    prevalence = float(outcome.iloc[:, 0].mean())
    null_ll = float(
        np.sum(
            outcome.iloc[:, 0] * math.log(prevalence)
            + (1 - outcome.iloc[:, 0]) * math.log(1 - prevalence)
        )
    )
    keep = design.columns != "Intercept"
    coefficients = fitted.coef[keep]
    standard_errors = fitted.se[keep]
    odds = np.exp(coefficients)
    return pd.DataFrame(
        {
            "termino": design.columns[keep],
            "coef": odds,
            "ee": odds * standard_errors,
            "ic_inf": np.exp(coefficients - 1.96 * standard_errors),
            "ic_sup": np.exp(coefficients + 1.96 * standard_errors),
            "p": p_values[keep],
            "n": len(data),
            "pseudo_r2": 1 - ordinary_ll / null_ll,
            "ee_tipo": "modelo",
        },
        columns=LOGIT_COLUMNS,
    )


def fit_regressions(
    transitions: pd.DataFrame,
    panel: pd.DataFrame,
    copper_prices: pd.DataFrame,
) -> RegressionTables:
    """Fit both specified associative models."""
    extended = prepare_extended_regression_data(transitions, panel, copper_prices)
    return RegressionTables(
        ols=fit_ols(transitions),
        ols_ampliado=fit_extended_ols(extended),
        logit=fit_logit(extended),
    )


def fit_regressions_from_parquet(
    path: Path,
    panel_path: Path,
    copper_prices_path: Path,
) -> RegressionTables:
    """Read transition output and fit both specified models."""
    return fit_regressions(
        pd.read_parquet(path),
        pd.read_parquet(panel_path),
        pd.read_parquet(copper_prices_path),
    )


def write_regressions(
    tables: RegressionTables, interim_dir: Path
) -> tuple[Path, Path, Path]:
    """Write both model tables."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    ols_path = interim_dir / "regresion_ols.parquet"
    logit_path = interim_dir / "regresion_logit.parquet"
    extended_path = interim_dir / "regresion_ols_ampliado.parquet"
    tables.ols.to_parquet(ols_path, index=False)
    tables.ols_ampliado.to_parquet(extended_path, index=False)
    tables.logit.to_parquet(logit_path, index=False)
    return ols_path, extended_path, logit_path
