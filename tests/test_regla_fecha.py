import pandas as pd

from cpdrift.conciliacion import build_date_rule_check


def test_date_rule_confusion_matrix_uses_declared_reason() -> None:
    panel = pd.DataFrame(
        [
            {"project_id": "A", "edicion": 2020, "proyecto_raw": "A", "puesta_en_marcha": 2021},
            {"project_id": "B", "edicion": 2020, "proyecto_raw": "B", "puesta_en_marcha": 2025},
            {"project_id": "C", "edicion": 2020, "proyecto_raw": "C", "puesta_en_marcha": 2020},
        ]
    )
    exits = pd.DataFrame(
        [
            [2020, 2021, "A", "operacion"],
            [2020, 2021, "B", "reestructuracion"],
            [2020, 2021, "C", "reestructuracion"],
        ],
        columns=["edicion_desde", "edicion_hasta", "proyecto_raw", "motivo"],
    )

    result = build_date_rule_check(panel, exits)

    matrix = result.matriz.set_index(["declarado", "regla"])["n"].to_dict()
    assert matrix[("operacion", "operacion")] == 1
    assert matrix[("reestructuracion", "reestructuracion")] == 1
    assert matrix[("reestructuracion", "operacion")] == 1
    assert result.detalle["discrepante"].sum() == 1


def test_date_rule_accepts_no_comparable_exits() -> None:
    exits = pd.DataFrame(
        [[2020, 2021, "A", "metodologia"]],
        columns=["edicion_desde", "edicion_hasta", "proyecto_raw", "motivo"],
    )

    result = build_date_rule_check(pd.DataFrame(), exits)

    assert result.matriz.empty
    assert result.detalle.empty
    assert set(result.matriz.columns) == {"declarado", "regla", "n"}
