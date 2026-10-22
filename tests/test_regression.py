import math

import pandas as pd
import pytest

from cpdrift.regression import fit_logit, fit_ols
from cpdrift.verify import recalculate_ols


def synthetic_transitions() -> pd.DataFrame:
    conditions = ["BASE", "PROBABLE", "POSIBLE", "POTENCIAL"]
    sectors = ["GRAN_MINERIA_CU", "Oro", "LITIO"]
    condition_terms = {"BASE": 0.0, "PROBABLE": 0.5, "POSIBLE": 1.0, "POTENCIAL": 1.5}
    sector_terms = {"GRAN_MINERIA_CU": 0.0, "Oro": 2.0, "LITIO": -1.0}
    rows: list[dict[str, object]] = []
    for index in range(36):
        condition = conditions[index % len(conditions)]
        sector = sectors[(index // 2) % len(sectors)]
        log_capex = 3.0 + index / 20
        horizon = 1 + (index % 7)
        rate = (
            1.0
            + condition_terms[condition]
            + sector_terms[sector]
            + 0.7 * log_capex
            + 0.2 * horizon
        )
        rows.append(
            {
                "project_id": f"P{index // 2:02d}",
                "ed_prev": 2020,
                "pem_prev": 2020 + horizon,
                "delta_pem": rate,
                "tasa_deriva": rate,
                "capex_prev_real": math.exp(log_capex),
                "condicion_prev": condition,
                "sector": sector,
            }
        )
    return pd.DataFrame(rows)


def test_ols_recovers_known_coefficients_without_noise() -> None:
    result = fit_ols(synthetic_transitions()).set_index("termino")

    assert result.loc["Intercept", "coef"] == pytest.approx(1.0)
    assert result.loc[
        'C(condicion_prev, Treatment("BASE"))[T.POTENCIAL]', "coef"
    ] == pytest.approx(1.5)
    assert result.loc["log_capex", "coef"] == pytest.approx(0.7)
    assert result.loc["horizonte", "coef"] == pytest.approx(0.2)


def test_independent_lstsq_matches_statsmodels() -> None:
    transitions = synthetic_transitions()
    fitted = fit_ols(transitions).set_index("termino")["coef"].to_dict()
    independent = recalculate_ols(transitions)

    assert independent["ols_coef_potencial"] == pytest.approx(
        fitted['C(condicion_prev, Treatment("BASE"))[T.POTENCIAL]'], abs=1e-9
    )
    assert independent["ols_coef_log_capex"] == pytest.approx(
        fitted["log_capex"], abs=1e-9
    )
    assert independent["ols_coef_horizonte"] == pytest.approx(
        fitted["horizonte"], abs=1e-9
    )
    assert independent["ols_n"] == len(transitions)


def test_logit_reports_odds_ratios_and_requested_columns() -> None:
    transitions = synthetic_transitions()
    transitions["delta_pem"] = [1 if index % 5 in {0, 1} else -1 for index in range(len(transitions))]

    result = fit_logit(transitions)

    assert set(result.columns) == {
        "termino",
        "coef",
        "ee",
        "ic_inf",
        "ic_sup",
        "p",
        "n",
        "pseudo_r2",
    }
    assert result["coef"].gt(0).all()
    assert result["n"].eq(len(transitions)).all()
