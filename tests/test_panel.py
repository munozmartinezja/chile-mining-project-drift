import pandas as pd

from cpdrift.panel import build_projects


def appearance(project_id: str, edition: int, pem: int) -> dict[str, object]:
    return {
        "project_id": project_id,
        "edicion": edition,
        "puesta_en_marcha": pem,
        "inversion_mmusd": float(edition),
        "proyecto_raw": project_id,
        "nombre_norm": project_id,
    }


def test_all_outcomes_and_2022_is_not_a_gap() -> None:
    panel = pd.DataFrame(
        [
            appearance("P001", 2024, 2030),
            appearance("P001", 2025, 2030),
            appearance("P002", 2020, 2021),
            appearance("P003", 2020, 2024),
            appearance("P004", 2021, 2025),
            appearance("P004", 2023, 2025),
            appearance("P005", 2016, 2020),
            appearance("P005", 2019, 2020),
        ]
    )
    exits = pd.DataFrame(
        [
            {
                "edicion_desde": edition,
                "proyecto_raw": project_id,
                "motivo": reason,
                "fuente": source,
            }
            for project_id, edition, reason, source in (
                ("P002", 2020, "operacion", "nombre"),
                ("P003", 2020, "reestructuracion", "cuadro"),
                ("P004", 2023, "metodologia", "revision"),
                ("P005", 2019, "fusion", "revision"),
            )
        ]
    )
    projects = build_projects(panel, exits).set_index("project_id")
    assert projects.loc["P001", "desenlace"] == "vigente"
    assert projects.loc["P002", "desenlace"] == "entro_en_operacion"
    assert projects.loc["P003", "desenlace"] == "salio_sin_operar"
    assert not bool(projects.loc["P004", "reingreso"])
    assert projects.loc["P004", "desenlace"] == "censurado"
    assert bool(projects.loc["P005", "reingreso"])
    assert projects.loc["P002", "motivo_salida"] == "operacion"
    assert projects.loc["P002", "fuente_desenlace"] == "nombre"
    assert projects.loc["P001", "motivo_salida"] == ""
    assert projects.loc["P001", "fuente_desenlace"] == "vigente"
