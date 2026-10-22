import ast
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cpdrift.claims import build_claims_register
from cpdrift.indicators import build_indicator_tables
from cpdrift.regression import fit_ols
from cpdrift.survival import build_cif_table
from cpdrift.verify import require_verified, verify_claims_register


def test_claims_and_independent_verification_agree(tmp_path: Path) -> None:
    panel = pd.DataFrame(
        [
            {
                "project_id": "P1",
                "edicion": 2020,
                "nombre_norm": "P1",
                "puesta_en_marcha": 2021,
                "inversion_mmusd": 100.0,
                "condicion": "POSIBLE",
                "sector": "COBRE",
                "region": "Atacama",
            },
            {
                "project_id": "P1",
                "edicion": 2021,
                "nombre_norm": "P1",
                "puesta_en_marcha": 2022,
                "inversion_mmusd": 120.0,
                "condicion": "PROBABLE",
                "sector": "COBRE",
                "region": "Atacama",
            },
            {
                "project_id": "A1",
                "edicion": 2021,
                "nombre_norm": "OTROS PROYECTOS",
                "puesta_en_marcha": 2025,
                "inversion_mmusd": 500.0,
                "condicion": "POSIBLE",
                "sector": "COBRE",
                "region": "Atacama",
            },
            {
                "project_id": "A2",
                "edicion": 2021,
                "nombre_norm": "PROYECTOS FURE",
                "puesta_en_marcha": 2025,
                "inversion_mmusd": 500.0,
                "condicion": "POSIBLE",
                "sector": "COBRE",
                "region": "Atacama",
            },
            {
                "project_id": "A3",
                "edicion": 2021,
                "nombre_norm": "TRANQUES",
                "puesta_en_marcha": 2025,
                "inversion_mmusd": 500.0,
                "condicion": "POSIBLE",
                "sector": "COBRE",
                "region": "Atacama",
            },
        ]
    )
    projects = pd.DataFrame(
        [
            {"project_id": "P1", "desenlace": "entro_en_operacion", "reingreso": False},
            {"project_id": "A1", "desenlace": "vigente", "reingreso": False},
            {"project_id": "A2", "desenlace": "vigente", "reingreso": False},
            {"project_id": "A3", "desenlace": "vigente", "reingreso": False},
        ]
    )
    cpi_path = tmp_path / "cpi.csv"
    pd.DataFrame(
        {"anio": [2020, 2021, 2025], "s1": [100, 110, 200], "s2": [100, 110, 200]}
    ).to_csv(cpi_path, index=False)
    panel_path = tmp_path / "panel.parquet"
    projects_path = tmp_path / "proyectos.parquet"
    panel.to_parquet(panel_path, index=False)
    projects.to_parquet(projects_path, index=False)

    bundle = build_indicator_tables(panel, projects, cpi_path)
    claims = build_claims_register(bundle)
    verified = verify_claims_register(claims, panel_path, projects_path, cpi_path)

    assert {item["estado"] for item in verified} == {"verificada"}
    assert next(
        item["valor"]
        for item in verified
        if item["id"] == "n_filas_agregadas_excluidas"
    ) == 3
    require_verified(verified)


def test_outcome_claims_are_recalculated_from_panel_and_references(
    tmp_path: Path,
) -> None:
    rows = []
    for project_id, edition in (
        ("O", 2020),
        ("R", 2020),
        ("C", 2020),
        ("S", 2020),
        ("S", 2021),
        ("N", 2021),
        ("S", 2025),
        ("N", 2025),
    ):
        rows.append(
            {
                "project_id": project_id,
                "edicion": edition,
                "proyecto_raw": project_id,
                "nombre_norm": project_id,
                "puesta_en_marcha": 2030,
                "inversion_mmusd": 100.0,
                "condicion": "BASE",
                "sector": "GRAN_MINERIA_CU",
                "region": "Atacama",
            }
        )
    panel = pd.DataFrame(rows)
    projects = pd.DataFrame(
        [
            {"project_id": "O", "desenlace": "entro_en_operacion", "reingreso": False},
            {"project_id": "R", "desenlace": "salio_sin_operar", "reingreso": False},
            {"project_id": "C", "desenlace": "censurado", "reingreso": False},
            {"project_id": "S", "desenlace": "vigente", "reingreso": False},
            {"project_id": "N", "desenlace": "vigente", "reingreso": False},
        ]
    )
    egresos = pd.DataFrame(
        [
            [2020, 2021, "O", "operacion", "nombre"],
            [2020, 2021, "R", "reestructuracion", "cuadro"],
            [2020, 2021, "C", "metodologia", "revision"],
        ],
        columns=[
            "edicion_desde",
            "edicion_hasta",
            "proyecto_raw",
            "motivo",
            "fuente",
        ],
    )
    reconciliation = pd.DataFrame(
        [
            {
                "edicion_desde": 2020,
                "edicion_hasta": 2021,
                "categoria": "nuevos",
                "estado": "ok",
            },
            {
                "edicion_desde": 2021,
                "edicion_hasta": 2025,
                "categoria": "nuevos",
                "estado": "ok",
            },
        ]
    )
    cpi_path = tmp_path / "cpi.csv"
    pd.DataFrame(
        {"anio": [2020, 2021, 2025], "s1": [100, 100, 100], "s2": [100, 100, 100]}
    ).to_csv(cpi_path, index=False)
    panel_path = tmp_path / "panel.parquet"
    projects_path = tmp_path / "proyectos.parquet"
    egresos_path = tmp_path / "egresos.csv"
    panel.to_parquet(panel_path, index=False)
    projects.to_parquet(projects_path, index=False)
    egresos.to_csv(egresos_path, index=False)

    indicators = build_indicator_tables(panel, projects, cpi_path, egresos)
    claims = build_claims_register(
        indicators,
        projects=projects,
        egresos=egresos,
        conciliacion=reconciliation,
    )
    verified = verify_claims_register(
        claims,
        panel_path,
        projects_path,
        cpi_path,
        egresos_path,
    )
    values = {item["id"]: item["valor"] for item in verified}

    assert {item["estado"] for item in verified} == {"verificada"}
    assert values["n_desenlace_operacion"] == 1
    assert values["n_desenlace_sin_operar"] == 1
    assert values["n_desenlace_censurado"] == 1
    assert values["pct_salidas_confirmadas_por_nombre"] == pytest.approx(100 / 3)
    assert values["n_pares_conciliados"] == 2
    assert values["n_salidas_umbral90_operacion"] == 1
    assert values["n_salidas_umbral90_sin_operar"] == 1

    changed = [dict(item) for item in claims]
    changed[0]["valor"] += 1
    failed = verify_claims_register(
        changed,
        panel_path,
        projects_path,
        cpi_path,
        egresos_path,
    )
    assert failed[0]["estado"] == "fallida"
    with pytest.raises(ValueError, match="cifras no verificadas"):
        require_verified(failed)


def test_verifier_does_not_import_indicator_or_claim_modules() -> None:
    source = Path("src/cpdrift/verify.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)

    assert "indicators" not in imports
    assert "claims" not in imports
    assert "cpdrift.indicators" not in imports
    assert "cpdrift.claims" not in imports
    assert "survival" not in imports
    assert "regression" not in imports
    assert "statsmodels" not in imports
    assert "cpdrift.survival" not in imports
    assert "cpdrift.regression" not in imports


def test_capex_claims_separate_unchanged_and_updated_transitions(tmp_path: Path) -> None:
    panel = pd.DataFrame(
        [
            {
                "project_id": "P1",
                "edicion": edition,
                "nombre_norm": "P1",
                "puesta_en_marcha": 2030,
                "inversion_mmusd": capex,
                "condicion": "BASE",
                "sector": "GRAN_MINERIA_CU",
                "region": "Atacama",
            }
            for edition, capex in ((2020, 100.0), (2021, 100.0), (2023, 200.0))
        ]
    )
    projects = pd.DataFrame(
        [{"project_id": "P1", "desenlace": "vigente", "reingreso": False}]
    )
    cpi_path = tmp_path / "cpi.csv"
    pd.DataFrame(
        {
            "anio": [2020, 2021, 2023, 2025],
            "s1": [100, 100, 100, 100],
            "s2": [100, 100, 100, 100],
        }
    ).to_csv(cpi_path, index=False)

    claims = {
        item["id"]: item["valor"]
        for item in build_claims_register(
            build_indicator_tables(panel, projects, cpi_path)
        )
    }

    assert claims["pct_capex_sin_actualizar"] == 50
    assert claims["mediana_delta_capex_nom_pct_actualizados"] == 100
    assert claims["mediana_delta_capex_real_pct_actualizados"] == 100
    assert "mediana_delta_capex_real_pct" not in claims


def test_model_claims_are_independently_verified(tmp_path: Path) -> None:
    conditions = ["BASE", "PROBABLE", "POSIBLE", "POTENCIAL"]
    sectors = ["GRAN_MINERIA_CU", "Oro", "LITIO"]
    rows: list[dict[str, object]] = []
    project_rows: list[dict[str, object]] = []
    for index in range(36):
        project_id = f"P{index:02d}"
        condition = conditions[index % 4]
        sector = sectors[(index // 2) % 3]
        capex = 100.0 + index * 3
        horizon = 1 + index % 7
        drift = (index % 5) - 1
        rows.extend(
            [
                {
                    "project_id": project_id,
                    "edicion": 2020,
                    "nombre_norm": project_id,
                    "puesta_en_marcha": 2020 + horizon,
                    "inversion_mmusd": capex,
                    "condicion": condition,
                    "sector": sector,
                    "region": "Atacama",
                },
                {
                    "project_id": project_id,
                    "edicion": 2021,
                    "nombre_norm": project_id,
                    "puesta_en_marcha": 2020 + horizon + drift,
                    "inversion_mmusd": capex + 10,
                    "condicion": condition,
                    "sector": sector,
                    "region": "Atacama",
                },
            ]
        )
        project_rows.append(
            {
                "project_id": project_id,
                "desenlace": (
                    "entro_en_operacion" if index % 2 == 0 else "salio_sin_operar"
                ),
                "reingreso": False,
            }
        )
    panel = pd.DataFrame(rows)
    projects = pd.DataFrame(project_rows)
    cpi_path = tmp_path / "cpi.csv"
    pd.DataFrame(
        {"anio": [2020, 2021, 2025], "s1": [100, 100, 100], "s2": [100, 100, 100]}
    ).to_csv(cpi_path, index=False)
    panel_path = tmp_path / "panel.parquet"
    projects_path = tmp_path / "proyectos.parquet"
    panel.to_parquet(panel_path, index=False)
    projects.to_parquet(projects_path, index=False)
    indicators = build_indicator_tables(panel, projects, cpi_path)
    cif = build_cif_table(indicators.proyectos_deriva, bootstrap_replicates=5)
    ols = fit_ols(indicators.transiciones)
    logit = pd.DataFrame(
        [
            {
                "termino": 'C(condicion_prev, Treatment("BASE"))[T.POSIBLE]',
                "coef": 1.25,
                "ee": 0.1,
                "ic_inf": 1.0,
                "ic_sup": 1.5,
                "p": 0.2,
                "n": 36,
                "pseudo_r2": 0.05,
            }
        ]
    )

    claims = build_claims_register(indicators, cif=cif, ols=ols, logit=logit)
    verified = verify_claims_register(claims, panel_path, projects_path, cpi_path)
    model_ids = {
        "n_cohorte_incidente",
        "cif_op_6_base_probable",
        "cif_sal_6_base_probable",
        "cif_op_6_posible_potencial",
        "cif_sal_6_posible_potencial",
        "ols_coef_potencial",
        "ols_coef_log_capex",
        "ols_coef_horizonte",
        "ols_n",
    }

    assert {
        item["estado"] for item in verified if item["id"] in model_ids
    } == {"verificada"}
    informative = [item for item in verified if item["estado"] == "informativa"]
    assert informative
    assert any(
        item["id"]
        == "logit_or_c_condicion_prev_treatment_base_t_posible"
        for item in informative
    )
    assert all(item["id"] != "logit_or_intercept" for item in informative)
    require_verified(verified)


def test_robustness_claims_are_registered() -> None:
    tables = SimpleNamespace(
        transiciones=pd.DataFrame(
            columns=[
                "delta_pem",
                "tasa_deriva",
                "capex_prev_nom",
                "capex_next_nom",
                "delta_capex_nom_pct",
                "delta_capex_real_pct",
            ]
        ),
        proyectos_deriva=pd.DataFrame(),
        entradas_salidas=pd.DataFrame(
            [{"salidas_sin_operar": 0, "salidas_operacion": 0}]
        ),
        fiabilidad={
            "pct_a_tiempo": None,
            "pct_hasta_1_anio": None,
            "mediana_deriva_acumulada": None,
        },
        n_filas_agregadas_excluidas=0,
    )

    potential = 'C(condicion_prev, Treatment("BASE"))[T.POTENCIAL]'
    ols_extended = pd.DataFrame(
        [
            {"termino": potential, "coef": 1.0, "p": 0.01},
            {"termino": "log_precio_cu", "coef": 2.0, "p": 0.02},
            {"termino": "estatal", "coef": 3.0, "p": 0.03},
        ]
    )
    validation = pd.DataFrame(
        [
            {
                "modelo": model,
                "auc": 0.7,
                "brier": 0.2,
                "calibracion_global": 0.01,
                "n_train": 10,
                "n_test": 5,
            }
            for model in ("completo", "solo_condicion", "prevalencia")
        ]
    )
    sensitivity = pd.DataFrame(
        [
            {
                "escenario": scenario,
                "modelo": "ols_ampliado",
                "metrica": metric,
                "valor": value,
                "n": 10,
            }
            for scenario in ("principal", "sin_par_2021_2023", "sin_estatales")
            for metric, value in ((f"coef:{potential}", 1.0), (f"p:{potential}", 0.01))
        ]
    )

    claims = build_claims_register(
        tables,
        ols_ampliado=ols_extended,
        validacion=validation,
        sensibilidad=sensitivity,
        regla_fecha=pd.DataFrame({"discrepante": [True, False]}),
    )
    values = {item["id"]: item["valor"] for item in claims}

    assert values["ols_amp_coef_precio_cu"] == 2.0
    assert values["auc_test_completo"] == 0.7
    assert values["n_escenarios_potencial_robusto"] == 3
    assert values["pct_regla_fecha_discrepante"] == 50
