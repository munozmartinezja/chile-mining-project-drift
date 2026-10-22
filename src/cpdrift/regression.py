"""Associative regression models for schedule drift transitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

FORMULA_TERMS = (
    'C(condicion_prev, Treatment("BASE"))'
    ' + C(grupo_sector, Treatment("COBRE"))'
    " + log_capex + horizonte"
)
OLS_FORMULA = f"tasa_deriva ~ {FORMULA_TERMS}"
LOGIT_FORMULA = f"atraso ~ {FORMULA_TERMS}"
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
]


@dataclass(frozen=True)
class RegressionTables:
    """Tabular OLS and logistic model results."""

    ols: pd.DataFrame
    logit: pd.DataFrame


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


def fit_logit(transitions: pd.DataFrame) -> pd.DataFrame:
    """Fit the associative binary model and report odds ratios."""
    data = prepare_regression_data(transitions)
    if data.empty or data["atraso"].nunique() < 2:
        raise ValueError("se requieren ambos valores de atraso para logit")
    fitted = smf.logit(LOGIT_FORMULA, data=data).fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": data["project_id"]},
    )
    intervals = fitted.conf_int()
    odds = np.exp(fitted.params)
    odds_intervals = np.exp(intervals)
    odds_standard_errors = odds * fitted.bse
    return pd.DataFrame(
        {
            "termino": fitted.params.index,
            "coef": odds.to_numpy(dtype=float),
            "ee": odds_standard_errors.to_numpy(dtype=float),
            "ic_inf": odds_intervals.iloc[:, 0].to_numpy(dtype=float),
            "ic_sup": odds_intervals.iloc[:, 1].to_numpy(dtype=float),
            "p": fitted.pvalues.to_numpy(dtype=float),
            "n": int(fitted.nobs),
            "pseudo_r2": float(fitted.prsquared),
        },
        columns=LOGIT_COLUMNS,
    )


def fit_regressions(transitions: pd.DataFrame) -> RegressionTables:
    """Fit both specified associative models."""
    return RegressionTables(ols=fit_ols(transitions), logit=fit_logit(transitions))


def fit_regressions_from_parquet(path: Path) -> RegressionTables:
    """Read transition output and fit both specified models."""
    return fit_regressions(pd.read_parquet(path))


def write_regressions(tables: RegressionTables, interim_dir: Path) -> tuple[Path, Path]:
    """Write both model tables."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    ols_path = interim_dir / "regresion_ols.parquet"
    logit_path = interim_dir / "regresion_logit.parquet"
    tables.ols.to_parquet(ols_path, index=False)
    tables.logit.to_parquet(logit_path, index=False)
    return ols_path, logit_path
