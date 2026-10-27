from __future__ import annotations

from pathlib import Path

import pandas as pd

from cpdrift.sensibilidad_publicable import build_publishable_sensitivity

SCENARIOS = [
    "principal",
    "censurados_como_salida",
    "censurados_como_operacion",
    "cuadre_como_operacion",
    "cuadre_como_salida",
    "sin_vinculos_debiles",
    "capex_inicial_ge_90",
]


def _synthetic_inputs(tmp_path: Path) -> tuple[object, ...]:
    conditions = ["BASE", "PROBABLE", "POSIBLE", "POTENCIAL"]
    sectors = ["GRAN_MINERIA_CU", "ORO", "MINERIA_INDUSTRIAL"]
    panel_rows = []
    transition_rows = []
    project_rows = []
    exit_rows = []
    for index in range(16):
        project_id = f"P{index + 1:03d}"
        name = f"PROYECTO {chr(65 + index)}"
        condition = conditions[index % len(conditions)]
        sector = sectors[index % len(sectors)]
        initial_capex = 80.0 if index == 0 else 100.0 + 10 * index
        initial_pem = 2026 + index % 5
        delta = index % 4
        for edition, pem, capex in (
            (2019, initial_pem, initial_capex),
            (2020, initial_pem + delta, initial_capex + 5),
        ):
            panel_rows.append(
                {
                    "project_id": project_id,
                    "edicion": edition,
                    "proyecto_raw": name,
                    "nombre_norm": name,
                    "operador_norm": f"OPERADOR {chr(65 + index)}",
                    "puesta_en_marcha": pem,
                    "inversion_mmusd": capex,
                    "condicion": condition,
                    "sector": sector,
                    "region": "Atacama",
                }
            )
        transition_rows.append(
            {
                "project_id": project_id,
                "ed_prev": 2019,
                "ed_next": 2020,
                "pem_prev": initial_pem,
                "delta_pem": float(delta),
                "tasa_deriva": float(delta),
                "capex_prev_real": initial_capex,
                "condicion_prev": condition,
                "sector": sector,
            }
        )
        if index == 3:
            outcome, reason = "censurado", "no_declarado"
        elif index % 2:
            outcome, reason = "salio_sin_operar", "reestructuracion"
        else:
            outcome, reason = "entro_en_operacion", "operacion"
        source = "cuadro" if index == 1 else "nombre"
        project_rows.append(
            {
                "project_id": project_id,
                "primera_edicion": 2019,
                "ultima_edicion": 2020,
                "desenlace": outcome,
                "fuente_desenlace": source,
            }
        )
        exit_rows.append(
            {
                "edicion_desde": 2020,
                "edicion_hasta": 2021,
                "proyecto_raw": name,
                "motivo": reason,
                "fuente": source,
            }
        )
    panel = pd.DataFrame(panel_rows)
    normalized = panel.drop(columns="project_id")
    manual_path = tmp_path / "cruce_manual.csv"
    pd.DataFrame(
        columns=[
            "edicion",
            "proyecto_raw",
            "decision",
            "destino_edicion",
            "destino_proyecto_raw",
            "nota",
            "evidencia",
        ]
    ).to_csv(manual_path, index=False)
    cpi_path = tmp_path / "cpi.csv"
    pd.DataFrame(
        {
            "anio": [2019, 2020, 2025],
            "s1": [100.0, 100.0, 100.0],
            "s2": [100.0, 100.0, 100.0],
        }
    ).to_csv(cpi_path, index=False)
    return (
        panel,
        pd.DataFrame(project_rows),
        pd.DataFrame(transition_rows),
        normalized,
        pd.DataFrame(exit_rows),
        cpi_path,
        manual_path,
    )


def test_publishable_sensitivity_scenarios_and_direction(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path)
    result = build_publishable_sensitivity(
        *inputs,
        interim_dir=tmp_path / "interim",
    ).set_index("escenario", drop=False)

    assert result["escenario"].tolist() == SCENARIOS
    assert (
        result.loc["censurados_como_salida", "cif_sal_6_posible_potencial"]
        >= result.loc["principal", "cif_sal_6_posible_potencial"]
    )
    assert result.loc["principal", "n_proyectos"] == 16
    assert result.loc["capex_inicial_ge_90", "n_proyectos"] == 15
