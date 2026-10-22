import numpy as np
import pandas as pd
import pytest

from cpdrift.validacion import (
    auc_mann_whitney,
    build_successive_validation,
    build_temporal_validation,
    resample_project_predictions,
)


def test_auc_uses_average_ranks_for_ties() -> None:
    assert auc_mann_whitney([0, 0, 1, 1], [0.1, 0.4, 0.4, 0.9]) == pytest.approx(
        0.875
    )


def test_temporal_validation_metrics_and_partition() -> None:
    rows = []
    for index in range(40):
        train = index < 24
        rows.append(
            {
                "project_id": f"P{index}",
                "ed_prev": (
                    (2018 if index < 12 else 2019)
                    if train
                    else (2020 if index < 32 else 2021)
                ),
                "atraso": index % 2,
                "condicion_prev": "BASE" if index % 3 else "POTENCIAL",
                "grupo_sector": "COBRE",
                "log_capex": 4.0 + index / 100,
                "horizonte": 2 + index % 4,
                "estatal": index % 5 == 0,
                "log_precio_cu": {
                    2018: 7.9,
                    2019: 8.0,
                    2020: 8.1,
                    2021: 8.2,
                }[
                    (2018 if index < 12 else 2019)
                    if train
                    else (2020 if index < 32 else 2021)
                ],
            }
        )

    result = build_temporal_validation(pd.DataFrame(rows))

    assert set(result.metricas["modelo"]) == {
        "completo",
        "sin_precio",
        "solo_condicion",
        "prevalencia",
    }
    assert result.metricas["n_train"].eq(24).all()
    assert result.metricas["n_test"].eq(16).all()
    assert set(result.predicciones["ed_prev"]) == {2020, 2021}
    prevalence = result.metricas.set_index("modelo").loc["prevalencia"]
    expected_brier = ((result.predicciones["atraso"] - 0.5) ** 2).mean()
    assert prevalence["brier"] == pytest.approx(expected_brier)
    complete = result.metricas.set_index("modelo").loc["completo"]
    assert complete["calibracion_global"] == pytest.approx(
        result.predicciones["p_completo"].mean()
        - result.predicciones["atraso"].mean()
    )


def test_successive_cuts_use_only_prior_rows(monkeypatch) -> None:
    editions = [2017, 2017, 2018, 2018, 2019, 2019, 2020, 2020,
                2021, 2021, 2023, 2023, 2024, 2024]
    data = pd.DataFrame(
        {
            "project_id": range(len(editions)),
            "ed_prev": editions,
            "atraso": [0, 1] * (len(editions) // 2),
        }
    )
    seen: list[tuple[int, int]] = []

    def fake_predictions(train, test, terms):
        del terms
        seen.append((int(train["ed_prev"].max()), int(test["ed_prev"].min())))
        return np.repeat(0.6, len(test))

    monkeypatch.setattr("cpdrift.validacion._firth_predictions", fake_predictions)
    result = build_successive_validation(data, bootstrap_replicates=10)
    assert set(result.cortes["corte"]) == {2018, 2019, 2020, 2021, 2023, 2024}
    assert all(train_max < test_min for train_max, test_min in seen)
    assert pd.isna(
        result.agregada.set_index("modelo").loc["prevalencia", "auc"]
    )
    assert result.cortes.loc[
        result.cortes["modelo"].eq("prevalencia"), "auc"
    ].isna().all()


def test_project_resampling_keeps_each_projects_rows_together() -> None:
    predictions = pd.DataFrame(
        {
            "project_id": [1, 1, 2, 2, 2],
            "row": [10, 11, 20, 21, 22],
        }
    )
    sampled = resample_project_predictions(
        predictions, np.random.default_rng(20261022)
    )
    for draw in sampled["bootstrap_draw"].unique():
        rows = sampled.loc[sampled["bootstrap_draw"].eq(draw), "row"].tolist()
        assert rows in ([10, 11], [20, 21, 22])
