import numpy as np
import pandas as pd
import pytest

from cpdrift.descriptivo import (
    build_fan_table,
    build_reconciliation_matrix,
    build_sensitivity_potential,
    build_ten_table,
    enrich_trajectories,
)
from cpdrift.etiquetas import (
    CATEGORY_LABELS,
    CONDITION_LABELS,
    CONDITION_ORDER,
    GROUP_LABELS,
    OUTCOME_LABELS,
    OUTCOME_ORDER,
    SECTOR_LABELS,
    SOURCE_LABELS,
    add_export_labels,
)


def test_label_dictionaries_are_complete() -> None:
    assert set(SECTOR_LABELS) == {"GRAN_MINERIA_CU", "MEDIANA_MINERIA_CU", "ESTATAL_CU", "PLANTAS_METALURGICAS", "MINERIA_INDUSTRIAL", "Oro", "Hierro", "Litio"}
    assert set(CONDITION_LABELS) == set(CONDITION_ORDER) == {"BASE", "PROBABLE", "POSIBLE", "POTENCIAL"}
    assert set(OUTCOME_LABELS) == set(OUTCOME_ORDER) == {"entro_en_operacion", "salio_sin_operar", "censurado", "vigente"}
    assert set(SOURCE_LABELS) == {"nombre", "cuadro", "revision", "vigente"}
    assert set(GROUP_LABELS) == {"BASE_PROBABLE", "POSIBLE_POTENCIAL", "TOTAL"}
    assert set(CATEGORY_LABELS) == {"entraron_operacion", "reestructuracion", "nuevos", "metodologia"}


def test_labels_cover_export_domains_and_reject_unknown_values() -> None:
    dimension = pd.DataFrame(
        {
            "sector": ["GRAN_MINERIA_CU"],
            "condicion_inicial": ["BASE"],
            "desenlace": ["vigente"],
            "fuente_desenlace": ["nombre"],
        }
    )
    labeled = add_export_labels("dim_proyecto", dimension)
    assert labeled.loc[0, "sector_etiqueta"] == "Gran minería del cobre"
    assert labeled.loc[0, "condicion_etiqueta"] == "Base"
    assert labeled.loc[0, "condicion_orden"] == 1
    assert labeled.loc[0, "desenlace_etiqueta"] == "En cartera 2025"
    assert labeled.loc[0, "desenlace_orden"] == 4
    assert labeled.loc[0, "fuente_etiqueta"] == "Confirmado por nombre"
    with pytest.raises(ValueError, match="sin etiqueta"):
        add_export_labels("dim_proyecto", dimension.assign(sector="DESCONOCIDO"))


def test_group_export_labels_use_approved_capitalization() -> None:
    source = pd.DataFrame(
        {"grupo": ["BASE_PROBABLE", "POSIBLE_POTENCIAL", "TOTAL"]}
    )

    labeled = add_export_labels("cif", source)

    assert labeled["grupo_etiqueta"].tolist() == [
        "Base o Probable",
        "Posible o Potencial",
        "Total",
    ]


def test_trajectory_and_fan_use_distance_from_first_appearance() -> None:
    rows = []
    for project in range(11):
        rows.extend(
            [
                {"id_publico": f"C3-{project:03d}", "edicion": 2018, "pem": 2020},
                {
                    "id_publico": f"C3-{project:03d}",
                    "edicion": 2022,
                    "pem": 2020 + project,
                },
            ]
        )
    trajectories = enrich_trajectories(pd.DataFrame(rows))
    fan = build_fan_table(trajectories).set_index("anios_desde_primera")
    assert 0 in fan.index and 4 in fan.index
    assert fan.loc[4, "n_proyectos"] == 11
    assert fan.loc[4, "mediana"] == 5
    assert fan.loc[4, "p90"] == np.percentile(np.arange(11), 90)
    assert build_fan_table(trajectories.iloc[:18]).empty


def test_trajectory_uses_literal_first_launch_year() -> None:
    source = pd.DataFrame(
        {
            "id_publico": ["C3-001", "C3-001"],
            "edicion": [2020, 2021],
            "pem": [np.nan, 2024],
        }
    )
    result = enrich_trajectories(source)
    assert result["deriva_desde_primera"].isna().all()


def test_ten_table_rounds_and_rejects_negative_remainder() -> None:
    cif = pd.DataFrame(
        {
            "cohorte": ["incidente", "incidente"],
            "grupo": ["BASE_PROBABLE", "BASE_PROBABLE"],
            "t": [4, 6],
            "cif_op": [0.34, 0.51],
            "cif_sal": [0.24, 0.21],
        }
    )
    table = build_ten_table(cif)
    assert table.loc[table["horizonte"].eq(4), ["operan", "salen", "resto"]].iloc[0].tolist() == [3, 2, 5]
    with pytest.raises(ValueError, match="resto negativo"):
        build_ten_table(cif.assign(cif_op=0.8, cif_sal=0.5))


def test_sensitivity_potential_and_reconciliation_matrix() -> None:
    term = 'C(condicion_prev, Treatment("BASE"))[T.POTENCIAL]'
    sensitivity = pd.DataFrame(
        [
            {
                "escenario": scenario,
                "modelo": "ols_ampliado",
                "metrica": f"{metric}:{term}",
                "valor": value,
            }
            for scenario in ("principal", "sin_2022", "sin_estatales")
            for metric, value in (("coef", 1.2), ("p", 0.03))
        ]
    )
    potential = build_sensitivity_potential(sensitivity)
    assert list(potential.columns) == ["escenario", "escenario_etiqueta", "coef", "p"]
    assert len(potential) == 3

    reconciliation = pd.DataFrame(
        [
            {
                "edicion_desde": 2014 + pair,
                "edicion_hasta": 2015 + pair,
                "categoria": category,
                "estado": "ok",
            }
            for pair in range(10)
            for category in ("entraron_operacion", "reestructuracion", "nuevos")
        ]
    )
    matrix = build_reconciliation_matrix(reconciliation)
    assert len(matrix) == 30
    assert matrix.iloc[0]["par"] == "2014→2015"
    assert matrix["simbolo"].eq("✓").all()
