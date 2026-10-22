from pathlib import Path

import pandas as pd

from cpdrift.descriptivo import (
    add_public_ids,
    build_descriptive_tables,
    delta_class,
)


def _cpi(path: Path) -> None:
    pd.DataFrame(
        {"anio": [2020, 2021, 2025], "s1": [100, 100, 200], "s2": [100, 100, 200]}
    ).to_csv(path, index=False)


def test_delta_classes_cover_published_bands() -> None:
    values = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 10]
    assert [delta_class(value) for value in values] == [
        "<=-2",
        "<=-2",
        "-1",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        ">=6",
        ">=6",
    ]


def test_portfolio_capex_is_summed_within_each_edition(tmp_path: Path) -> None:
    cpi_path = tmp_path / "cpi.csv"
    _cpi(cpi_path)
    panel = pd.DataFrame(
        {
            "project_id": [1, 2, 1],
            "nombre_norm": ["A", "B", "A"],
            "edicion": [2020, 2020, 2021],
            "inversion_mmusd": [10.0, 20.0, 40.0],
            "puesta_en_marcha": [2022, 2023, 2024],
            "condicion": ["BASE", "PROBABLE", "BASE"],
            "sector": ["ORO", "ORO", "ORO"],
            "region": ["NORTE", "NORTE", "NORTE"],
        }
    )
    projects = pd.DataFrame(
        {
            "project_id": [1, 2],
            "primera_edicion": [2020, 2020],
            "ultima_edicion": [2021, 2020],
            "n_ediciones": [2, 1],
            "pem_primera": [2022, 2023],
            "capex_primero": [10.0, 20.0],
            "desenlace": ["vigente", "censurado"],
            "fuente_desenlace": ["vigente", "cuadro"],
            "reingreso": [False, False],
        }
    )
    transitions = pd.DataFrame(
        columns=["project_id", "delta_pem", "condicion_prev"]
    )
    drift = pd.DataFrame(
        {
            "project_id": [1, 2],
            "capex_inicial_real": [20.0, 40.0],
            "condicion_inicial": ["BASE", "PROBABLE"],
            "sector": ["ORO", "ORO"],
            "region": ["NORTE", "NORTE"],
            "desenlace": ["vigente", "censurado"],
        }
    )
    tables, _ = build_descriptive_tables(
        panel, projects, transitions, drift, pd.DataFrame(), cpi_path
    )
    portfolio = tables["desc_cartera_edicion"].set_index("edicion")
    assert portfolio.loc[2020, "capex_nominal_total"] == 30
    assert portfolio.loc[2021, "capex_nominal_total"] == 40
    assert portfolio.loc[2020, "capex_real_total"] == 60
    assert portfolio.loc[2021, "capex_real_total"] == 80


def test_public_ids_are_deterministic_under_input_reordering() -> None:
    projects = pd.DataFrame(
        {
            "project_id": [30, 10, 20],
            "primera_edicion": [2020, 2019, 2019],
            "pem_primera": [2024, 2025, 2024],
            "capex_primero": [50, 100, 200],
        }
    )
    first = add_public_ids(projects).set_index("project_id")["id_publico"]
    second = add_public_ids(projects.sample(frac=1, random_state=7)).set_index(
        "project_id"
    )["id_publico"]
    pd.testing.assert_series_equal(first.sort_index(), second.sort_index())
    assert first.to_dict() == {30: "C3-003", 10: "C3-002", 20: "C3-001"}
