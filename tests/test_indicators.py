from pathlib import Path

import pandas as pd
import pytest

from cpdrift.indicators import build_indicator_tables


def appearance(
    project_id: str,
    edition: int,
    *,
    name: str | None = None,
    pem: int = 2030,
    capex: float = 100.0,
    condition: str = "POSIBLE",
    sector: str = "COBRE",
    region: str = "Atacama",
) -> dict[str, object]:
    return {
        "project_id": project_id,
        "edicion": edition,
        "nombre_norm": name or project_id,
        "puesta_en_marcha": pem,
        "inversion_mmusd": capex,
        "condicion": condition,
        "sector": sector,
        "region": region,
    }


def project(
    project_id: str,
    outcome: str = "vigente",
    reentry: bool = False,
) -> dict[str, object]:
    return {"project_id": project_id, "desenlace": outcome, "reingreso": reentry}


def cpi_file(tmp_path: Path, values: dict[int, float]) -> Path:
    path = tmp_path / "cpi.csv"
    pd.DataFrame(
        {"anio": list(values), "s1": list(values.values()), "s2": list(values.values())}
    ).to_csv(path, index=False)
    return path


def test_aggregates_deflator_transitions_and_single_appearance(tmp_path: Path) -> None:
    panel = pd.DataFrame(
        [
            appearance("P1", 2021, pem=2025, capex=100),
            appearance("P1", 2023, pem=2027, capex=120),
            appearance("P2", 2023, pem=2031, capex=90),
            appearance("A1", 2023, name="OTROS PROYECTOS", capex=500),
            appearance("A2", 2023, name="CARTERA PROYECTOS DE INFORMACION", capex=600),
        ]
    )
    projects = pd.DataFrame([project("P1"), project("P2"), project("A1"), project("A2")])
    bundle = build_indicator_tables(
        panel,
        projects,
        cpi_file(tmp_path, {2021: 100, 2023: 150, 2025: 200}),
    )

    assert bundle.n_filas_agregadas_excluidas == 2
    transition = bundle.transiciones.iloc[0]
    assert transition["ed_prev"] == 2021
    assert transition["ed_next"] == 2023
    assert transition["gap_anios"] == 2
    assert transition["delta_pem"] == 2
    assert transition["tasa_deriva"] == 1
    assert transition["capex_prev_real"] == 200
    assert transition["capex_next_real"] == pytest.approx(160)

    single = bundle.proyectos_deriva.set_index("project_id").loc["P2"]
    assert single["n_apariciones"] == 1
    assert pd.isna(single["tasa_acumulada"])
    assert set(bundle.transiciones["project_id"]) == {"P1"}


def test_missing_cpi_year_is_an_error(tmp_path: Path) -> None:
    panel = pd.DataFrame([appearance("P1", 2024)])
    projects = pd.DataFrame([project("P1")])

    with pytest.raises(ValueError, match="CPI.*2024"):
        build_indicator_tables(panel, projects, cpi_file(tmp_path, {2025: 200}))


def test_reliability_for_operated_projects(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    refs: list[dict[str, object]] = []
    for project_id, drift in (("P0", 0), ("P1", 1), ("P3", 3)):
        rows.extend(
            [
                appearance(project_id, 2020, pem=2021),
                appearance(project_id, 2021, pem=2021 + drift),
            ]
        )
        refs.append(project(project_id, "entro_en_operacion"))

    bundle = build_indicator_tables(
        pd.DataFrame(rows),
        pd.DataFrame(refs),
        cpi_file(tmp_path, {2020: 100, 2021: 110, 2025: 200}),
    )

    assert bundle.fiabilidad["pct_a_tiempo"] == pytest.approx(100 / 3)
    assert bundle.fiabilidad["pct_hasta_1_anio"] == pytest.approx(200 / 3)
    assert bundle.fiabilidad["mediana_deriva_acumulada"] == 1


def test_entries_and_exits_use_homogeneous_threshold(tmp_path: Path) -> None:
    panel = pd.DataFrame(
        [
            appearance("A", 2020, pem=2021),
            appearance("B", 2020, pem=2030),
            appearance("R", 2020, pem=2030),
            appearance("S", 2020),
            appearance("LOW", 2020, capex=89),
            appearance("S", 2021),
            appearance("N", 2021),
            appearance("LOW", 2021, capex=89),
            appearance("S", 2023),
            appearance("N", 2023),
            appearance("R", 2023),
            appearance("N2", 2023),
        ]
    )
    ids = panel["project_id"].unique()
    projects = pd.DataFrame([project(value, reentry=value == "R") for value in ids])
    bundle = build_indicator_tables(
        panel,
        projects,
        cpi_file(tmp_path, {2020: 100, 2021: 110, 2023: 150, 2025: 200}),
    )

    first, second = bundle.entradas_salidas.to_dict("records")
    assert first == {
        "ed_prev": 2020,
        "ed_next": 2021,
        "n_base": 4,
        "salidas_operacion": 1,
        "salidas_sin_operar": 2,
        "entradas_nuevas": 1,
        "reingresos": 0,
    }
    assert second == {
        "ed_prev": 2021,
        "ed_next": 2023,
        "n_base": 2,
        "salidas_operacion": 0,
        "salidas_sin_operar": 0,
        "entradas_nuevas": 1,
        "reingresos": 1,
    }


def test_group_summaries_use_initial_condition(tmp_path: Path) -> None:
    panel = pd.DataFrame(
        [
            appearance("P1", 2020, pem=2022, condition="BASE"),
            appearance("P1", 2021, pem=2024, condition="CAMBIO"),
            appearance("P2", 2020, pem=2022, condition="BASE"),
            appearance("P2", 2021, pem=2021, condition="BASE"),
        ]
    )
    bundle = build_indicator_tables(
        panel,
        pd.DataFrame([project("P1"), project("P2")]),
        cpi_file(tmp_path, {2020: 100, 2021: 100, 2025: 100}),
    )

    summary = bundle.resumenes["condicion_inicial"].set_index("condicion_inicial")
    assert summary.loc["BASE", "n"] == 2
    assert summary.loc["BASE", "media_tasa_deriva"] == pytest.approx(0.5)
    assert summary.loc["BASE", "pct_transiciones_atraso"] == 50


def test_el_abra_anchor_is_in_manual_reference() -> None:
    manual = pd.read_csv("data/reference/cruce_manual.csv", keep_default_na=False)
    row = manual.loc[
        manual["edicion"].eq(2025)
        & manual["proyecto_raw"].eq("Continuidad Operacional Minera El Abra")
    ].iloc[0]

    assert row["edicion"] == 2025
    assert row["decision"] == "mismo"
    assert row["destino_edicion"] == "2024"
    assert row["destino_proyecto_raw"] == "Expansión El Abra"
