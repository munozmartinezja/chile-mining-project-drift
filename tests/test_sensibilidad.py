import pandas as pd

from cpdrift.sensibilidad import survival_scenarios


def test_only_name_confirmed_censors_table_outcomes() -> None:
    projects = pd.DataFrame(
        [
            {
                "project_id": "P1",
                "primera_edicion": 2018,
                "ultima_edicion": 2020,
                "desenlace": "entro_en_operacion",
                "fuente_desenlace": "cuadro",
                "condicion_inicial": "BASE",
            },
            {
                "project_id": "P2",
                "primera_edicion": 2018,
                "ultima_edicion": 2020,
                "desenlace": "entro_en_operacion",
                "fuente_desenlace": "nombre",
                "condicion_inicial": "BASE",
            },
            {
                "project_id": "P3",
                "primera_edicion": 2018,
                "ultima_edicion": 2025,
                "desenlace": "vigente",
                "fuente_desenlace": "vigente",
                "condicion_inicial": "BASE",
            },
        ]
    )
    panel = pd.DataFrame(
        [
            {"project_id": project_id, "edicion": 2018, "operador_norm": "PRIVADA"}
            for project_id in ("P1", "P2", "P3")
        ]
    )

    scenarios = survival_scenarios(projects, panel)
    selected = scenarios.loc[
        scenarios["escenario"].eq("solo_confirmados_por_nombre")
        & scenarios["grupo"].eq("BASE_PROBABLE")
        & scenarios["t"].eq(4)
    ].iloc[0]

    assert selected["cif_op"] == 0.5
