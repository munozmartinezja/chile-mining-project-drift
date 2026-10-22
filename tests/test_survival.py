from pathlib import Path

import pandas as pd
import pytest

from cpdrift.survival import (
    build_cif_table,
    estimate_cif,
    prepare_survival_endpoints,
)


def project(
    project_id: str,
    first: int,
    last: int,
    outcome: str,
    condition: str = "BASE",
) -> dict[str, object]:
    return {
        "project_id": project_id,
        "primera_edicion": first,
        "ultima_edicion": last,
        "desenlace": outcome,
        "condicion_inicial": condition,
    }


def test_cif_matches_hand_calculation_for_competing_events() -> None:
    endpoints = pd.DataFrame(
        {"t": [1, 1, 2, 2, 3, 3], "evento": [1, 2, 1, 2, 0, 0]}
    )

    curve = estimate_cif(endpoints).set_index("t")

    assert curve.loc[1, "en_riesgo"] == 6
    assert curve.loc[1, "cif_op"] == pytest.approx(1 / 6)
    assert curve.loc[1, "cif_sal"] == pytest.approx(1 / 6)
    assert curve.loc[2, "en_riesgo"] == 4
    assert curve.loc[2, "cif_op"] == pytest.approx(1 / 3)
    assert curve.loc[2, "cif_sal"] == pytest.approx(1 / 3)


def test_single_event_cif_matches_one_minus_kaplan_meier() -> None:
    endpoints = pd.DataFrame({"t": [1, 2, 3], "evento": [1, 1, 1]})

    curve = estimate_cif(endpoints)
    survival = 1.0
    for row in curve.itertuples(index=False):
        survival *= 1 - row.eventos_op / row.en_riesgo
        assert row.cif_op == pytest.approx(1 - survival)
        assert row.cif_sal == 0


def test_endpoint_rules_exclude_zero_and_use_next_data_edition() -> None:
    projects = pd.DataFrame(
        [
            project("ZERO", 2025, 2025, "vigente"),
            project("GAP", 2019, 2021, "entro_en_operacion"),
            project("CURRENT", 2020, 2025, "vigente"),
        ]
    )

    endpoints = prepare_survival_endpoints(projects).set_index("project_id")

    assert "ZERO" not in endpoints.index
    assert endpoints.loc["GAP", "t"] == 4
    assert endpoints.loc["GAP", "evento"] == 1
    assert endpoints.loc["CURRENT", "t"] == 5
    assert endpoints.loc["CURRENT", "evento"] == 0


def test_censored_outcome_is_censored_at_last_appearance() -> None:
    projects = pd.DataFrame(
        [project("CENSORED", 2018, 2021, "censurado")]
    )

    endpoint = prepare_survival_endpoints(projects).iloc[0]

    assert endpoint["t"] == 3
    assert endpoint["evento"] == 0


def test_incident_cohort_excludes_projects_starting_in_2014() -> None:
    projects = pd.DataFrame(
        [
            project("OLD", 2014, 2021, "salio_sin_operar"),
            project("NEW", 2015, 2021, "entro_en_operacion", "POSIBLE"),
        ]
    )

    table = build_cif_table(projects, bootstrap_replicates=10)
    incident = table.loc[
        table["cohorte"].eq("incidente") & table["grupo"].eq("TOTAL")
    ]

    assert incident["en_riesgo"].max() == 1


def test_bootstrap_is_deterministic_with_fixed_seed() -> None:
    projects = pd.DataFrame(
        [
            project("A", 2015, 2016, "entro_en_operacion"),
            project("B", 2015, 2017, "salio_sin_operar", "POSIBLE"),
            project("C", 2015, 2025, "vigente", "PROBABLE"),
            project("D", 2016, 2025, "vigente", "POTENCIAL"),
        ]
    )

    first = build_cif_table(projects, bootstrap_replicates=40)
    second = build_cif_table(projects, bootstrap_replicates=40)

    pd.testing.assert_frame_equal(first, second)


def test_bootstrap_changes_with_another_seed() -> None:
    projects = pd.DataFrame(
        {
            "project_id": range(12),
            "primera_edicion": [2015] * 12,
            "ultima_edicion": [2016] * 8 + [2025] * 4,
            "desenlace": ["entro_en_operacion"] * 4
            + ["salio_sin_operar"] * 4
            + ["vigente"] * 4,
            "condicion_inicial": ["BASE"] * 12,
        }
    )
    first = build_cif_table(projects, bootstrap_replicates=40, seed=20261021)
    second = build_cif_table(projects, bootstrap_replicates=40, seed=7)
    columns = ["ic_op_inf", "ic_op_sup", "ic_sal_inf", "ic_sal_sup"]
    assert not first[columns].equals(second[columns])


def test_write_source_does_not_require_pdf(tmp_path: Path) -> None:
    projects = pd.DataFrame([project("A", 2015, 2025, "vigente")])
    table = build_cif_table(projects, bootstrap_replicates=2)
    path = tmp_path / "cif.parquet"
    table.to_parquet(path, index=False)
    assert path.exists()
