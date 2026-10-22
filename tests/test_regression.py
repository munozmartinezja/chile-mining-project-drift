import math

import numpy as np
import pandas as pd
import pytest

from cpdrift.regression import (
    OLS_EXTENDED_FORMULA,
    fit_extended_ols,
    fit_logit,
    fit_ols,
    prepare_extended_regression_data,
    prepare_regression_data,
)
from cpdrift.verify import _clustered_ols, recalculate_ols


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

    data = prepare_regression_data(transitions)
    rng = np.random.default_rng(19)
    data["estatal"] = rng.binomial(1, 0.4, len(data))
    data["log_precio_cu"] = rng.normal(8, 0.2, len(data))
    result = fit_logit(data)

    assert set(result.columns) == {
        "termino",
        "coef",
        "ee",
        "ic_inf",
        "ic_sup",
        "p",
        "n",
        "pseudo_r2",
        "ee_tipo",
    }
    assert result["coef"].gt(0).all()
    assert result["n"].eq(len(transitions)).all()
    assert result["ee_tipo"].eq("modelo").all()
    assert "Intercept" not in set(result["termino"])


def test_extended_variables_use_previous_appearance_operator_and_price() -> None:
    transitions = synthetic_transitions().iloc[[0, 2]].copy()
    panel = pd.DataFrame(
        [
            {"project_id": transitions.iloc[0]["project_id"], "edicion": 2020, "operador_norm": "CODELCO Norte"},
            {"project_id": transitions.iloc[1]["project_id"], "edicion": 2020, "operador_norm": "Privada"},
        ]
    ).drop_duplicates(["project_id", "edicion"])
    prices = pd.DataFrame([{"edicion": 2020, "precio_real_2025": math.exp(8)}])

    result = prepare_extended_regression_data(transitions, panel, prices)

    assert result.iloc[0]["estatal"] == 1
    assert result.iloc[1]["estatal"] == 0
    assert result["log_precio_cu"].eq(8).all()


def test_independent_clustered_ols_matches_expanded_model() -> None:
    rng = np.random.default_rng(42)
    size = 200
    data = pd.DataFrame(
        {
            "project_id": [f"P{index // 2}" for index in range(size)],
            "tasa_deriva": rng.normal(size=size),
            "condicion_prev": rng.choice(
                ["BASE", "PROBABLE", "POSIBLE", "POTENCIAL"], size=size
            ),
            "grupo_sector": rng.choice(["COBRE", "ORO", "OTROS"], size=size),
            "log_capex": rng.normal(5, 1, size=size),
            "horizonte": rng.normal(4, 2, size=size),
            "estatal": rng.binomial(1, 0.3, size=size),
            "log_precio_cu": rng.normal(8, 0.2, size=size),
        }
    )

    expected = fit_extended_ols(data).set_index("termino")
    coefficients, p_values = _clustered_ols(data, OLS_EXTENDED_FORMULA)

    for term in ("estatal", "log_precio_cu"):
        assert coefficients[term] == pytest.approx(expected.loc[term, "coef"], abs=1e-9)
        assert p_values[term] == pytest.approx(expected.loc[term, "p"], abs=1e-9)
