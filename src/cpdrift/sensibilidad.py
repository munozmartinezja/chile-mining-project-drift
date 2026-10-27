"""Sensitivity scenarios for associative regressions and competing risks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

from .regression import OLS_EXTENDED_FORMULA
from .survival import build_cif_table


def regression_scenarios(data: pd.DataFrame) -> pd.DataFrame:
    """Fit expanded OLS under the three requested samples."""
    scenarios = {
        "principal": (data, OLS_EXTENDED_FORMULA),
        "sin_2022": (
            data.loc[~(data["ed_prev"].eq(2022) | data["ed_next"].eq(2022))],
            OLS_EXTENDED_FORMULA,
        ),
        "sin_estatales": (
            data.loc[data["estatal"].eq(0)],
            OLS_EXTENDED_FORMULA.replace(" + estatal", ""),
        ),
    }
    rows = []
    for scenario, (sample, formula) in scenarios.items():
        fitted = smf.ols(formula, data=sample).fit(
            cov_type="cluster", cov_kwds={"groups": sample["project_id"]}
        )
        for term in fitted.params.index:
            rows.extend(
                [
                    {
                        "escenario": scenario,
                        "modelo": "ols_ampliado",
                        "metrica": f"coef:{term}",
                        "valor": float(fitted.params[term]),
                        "n": int(fitted.nobs),
                    },
                    {
                        "escenario": scenario,
                        "modelo": "ols_ampliado",
                        "metrica": f"p:{term}",
                        "valor": float(fitted.pvalues[term]),
                        "n": int(fitted.nobs),
                    },
                ]
            )
    return pd.DataFrame(rows)


def survival_scenarios(projects: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Calculate requested competing-risk sensitivity scenarios."""
    first_operators = (
        panel.sort_values("edicion", kind="stable")
        .drop_duplicates("project_id")
        .set_index("project_id")["operador_norm"]
        .fillna("")
        .astype(str)
        .str.upper()
    )
    scenarios = {
        "principal": projects,
        "cohorte_total": projects,
        "sin_estatales": projects.loc[
            ~projects["project_id"]
            .map(first_operators)
            .fillna("")
            .str.contains(r"CODELCO|ENAMI", regex=True)
        ],
    }
    confirmed = projects.copy()
    mask = confirmed["fuente_desenlace"].eq("cuadro")
    confirmed.loc[mask, "desenlace"] = "censurado"
    scenarios["solo_confirmados_por_nombre"] = confirmed
    rows = []
    for scenario, sample in scenarios.items():
        table = build_cif_table(sample, bootstrap_replicates=0)
        cohort = "total" if scenario == "cohorte_total" else "incidente"
        selected = table.loc[
            table["cohorte"].eq(cohort)
            & table["grupo"].isin({"BASE_PROBABLE", "POSIBLE_POTENCIAL"})
            & table["t"].isin({4, 6})
        ]
        for row in selected.itertuples(index=False):
            rows.append(
                {
                    "escenario": scenario,
                    "grupo": row.grupo,
                    "t": int(row.t),
                    "cif_op": float(row.cif_op),
                    "cif_sal": float(row.cif_sal),
                    "en_riesgo": int(row.en_riesgo),
                }
            )
    return pd.DataFrame(rows)


def build_sensitivity(
    regression_data: pd.DataFrame,
    projects: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Return all regression and survival scenarios in long format."""
    regression = regression_scenarios(regression_data)
    survival = survival_scenarios(projects, panel)
    survival_rows = []
    for row in survival.itertuples(index=False):
        for metric in ("cif_op", "cif_sal", "en_riesgo"):
            survival_rows.append(
                {
                    "escenario": row.escenario,
                    "modelo": f"supervivencia:{row.grupo}:t{row.t}",
                    "metrica": metric,
                    "valor": float(getattr(row, metric)),
                    "n": int(row.en_riesgo),
                }
            )
    return pd.concat([regression, pd.DataFrame(survival_rows)], ignore_index=True)


def write_sensitivity(table: pd.DataFrame, interim_dir: Path) -> Path:
    """Write the long sensitivity table."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    path = interim_dir / "sensibilidad.parquet"
    table.to_parquet(path, index=False)
    return path
