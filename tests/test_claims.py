import ast
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cpdrift.claims import build_claims_register
from cpdrift.descriptivo import build_descriptive_tables
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
    assert all(item["descripcion"] != item["id"].replace("_", " ") for item in claims)
    assert all(item["unidad"] != "valor" for item in claims)
    combination_ids = {
        item["id"] for item in claims if item["unidad"] == "combinaciones"
    }
    assert combination_ids <= {
        "n_conciliaciones_exactas",
        "n_conciliaciones_con_ajuste",
    }
    verified = verify_claims_register(claims, panel_path, projects_path, cpi_path)

    assert {item["estado"] for item in verified} == {"verificada"}
    assert next(
        item["valor"]
        for item in verified
        if item["id"] == "n_filas_agregadas_excluidas"
    ) == 3
    require_verified(verified)


def test_reference_and_seen_project_claims_are_registered() -> None:
    tables = SimpleNamespace(
        transiciones=pd.DataFrame(
            columns=[
                "capex_prev_nom",
                "capex_next_nom",
                "delta_pem",
                "tasa_deriva",
                "delta_capex_nom_pct",
                "delta_capex_real_pct",
            ]
        ),
        fiabilidad=pd.Series(
            {
                "pct_a_tiempo": None,
                "pct_hasta_1_anio": None,
                "mediana_deriva_acumulada": None,
            }
        ),
        entradas_salidas=pd.DataFrame(
            {"salidas_sin_operar": [0], "salidas_operacion": [0]}
        ),
        proyectos_deriva=pd.DataFrame(),
        n_filas_agregadas_excluidas=0,
    )
    transcripcion = pd.DataFrame({"proyecto": ["A", "B", "C"]})
    manual = pd.DataFrame(
        {
            "decision": ["mismo", "mismo", "mismo", "nuevo"],
            "evidencia": [
                "operador_capex",
                "conciliacion",
                "nombre_comun",
                "conciliacion",
            ],
        }
    )
    egresos = pd.DataFrame({"fuente": ["cuadro", "nombre", "cuadro"]})
    validation_data = pd.DataFrame(
        {
            "project_id": ["A", "B", "A", "C"],
            "ed_prev": [2017, 2017, 2018, 2018],
        }
    )
    predictions = pd.DataFrame(
        {"project_id": ["A", "C"], "corte": [2018, 2018]}
    )

    claims = build_claims_register(
        tables,
        egresos=egresos,
        transcripcion_2022=transcripcion,
        cruce_manual=manual,
        validacion_datos=validation_data,
        validacion_predicciones=predictions,
    )
    values = {item["id"]: item["valor"] for item in claims}

    assert values["n_filas_transcritas_2022"] == 3
    assert values["n_vinculos_debiles"] == 2
    assert values["n_egresos_declarados"] == 3
    assert values["n_egresos_por_cuadro"] == 2
    assert values["pct_test_proyectos_vistos"] == 50


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
                "categoria": "entraron_operacion",
                "delta_n": 0,
                "delta_capex": 0.0,
                "compara_capex": True,
                "estado": "ok",
            },
            {
                "edicion_desde": 2020,
                "edicion_hasta": 2021,
                "categoria": "reestructuracion",
                "delta_n": 0,
                "delta_capex": 0.0,
                "compara_capex": True,
                "estado": "ok",
            },
            {
                "edicion_desde": 2020,
                "edicion_hasta": 2021,
                "categoria": "metodologia",
                "delta_n": 1,
                "delta_capex": 0.0,
                "compara_capex": True,
                "estado": "ok",
            },
            {
                "edicion_desde": 2020,
                "edicion_hasta": 2021,
                "categoria": "nuevos",
                "delta_n": 0,
                "delta_capex": 0.0,
                "compara_capex": True,
                "estado": "ok",
            },
            {
                "edicion_desde": 2021,
                "edicion_hasta": 2025,
                "categoria": "nuevos",
                "delta_n": 0,
                "delta_capex": 0.0,
                "compara_capex": True,
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
    cambios_path = tmp_path / "cambios.csv"
    ajustes_path = tmp_path / "ajustes.csv"
    transcripcion_path = tmp_path / "cartera_2022_transcrita.csv"
    manual_path = tmp_path / "cruce_manual.csv"
    panel.to_parquet(panel_path, index=False)
    projects.to_parquet(projects_path, index=False)
    egresos.to_csv(egresos_path, index=False)
    pd.DataFrame(
        [
            [2021, 2020, "entraron_operacion", 1, 100.0],
            [2021, 2020, "reestructuracion", 1, 100.0],
            [2021, 2020, "metodologia", 2, 100.0],
            [2021, 2020, "nuevos", 1, 100.0],
            [2025, 2021, "nuevos", 0, 0.0],
        ],
        columns=[
            "edicion",
            "par_desde",
            "categoria",
            "n_declarado",
            "capex_mmusd",
        ],
    ).to_csv(cambios_path, index=False)
    pd.DataFrame(
        [[2020, 2021, "metodologia", 1, 0.0]],
        columns=[
            "edicion_desde",
            "edicion_hasta",
            "categoria",
            "delta_n",
            "delta_capex",
        ],
    ).to_csv(ajustes_path, index=False)
    transcripcion = pd.DataFrame({"proyecto": ["A", "B"]})
    manual = pd.DataFrame(
        {"decision": ["mismo", "mismo"], "evidencia": ["conciliacion", "nombre_comun"]}
    )
    transcripcion.to_csv(transcripcion_path, index=False)
    manual.to_csv(manual_path, index=False)

    indicators = build_indicator_tables(panel, projects, cpi_path, egresos)
    claims = build_claims_register(
        indicators,
        projects=projects,
        egresos=egresos,
        conciliacion=reconciliation,
        transcripcion_2022=transcripcion,
        cruce_manual=manual,
    )
    verified = verify_claims_register(
        claims,
        panel_path,
        projects_path,
        cpi_path,
        egresos_path,
        cambios_path=cambios_path,
        ajustes_path=ajustes_path,
        transcripcion_2022_path=transcripcion_path,
        cruce_manual_path=manual_path,
    )
    values = {item["id"]: item["valor"] for item in verified}

    assert {item["estado"] for item in verified} == {"verificada"}
    assert values["n_desenlace_operacion"] == 1
    assert values["n_desenlace_sin_operar"] == 1
    assert values["n_desenlace_censurado"] == 1
    assert values["pct_salidas_confirmadas_por_nombre"] == pytest.approx(100 / 3)
    assert values["n_pares_conciliados"] == 2
    assert values["n_conciliaciones_exactas"] == 4
    assert values["n_conciliaciones_con_ajuste"] == 1
    assert values["n_salidas_umbral90_operacion"] == 1
    assert values["n_salidas_umbral90_sin_operar"] == 1
    assert values["n_filas_transcritas_2022"] == 2
    assert values["n_vinculos_debiles"] == 1
    assert values["n_egresos_declarados"] == 3
    assert values["n_egresos_por_cuadro"] == 1

    changed = [dict(item) for item in claims]
    changed[0]["valor"] += 1
    failed = verify_claims_register(
        changed,
        panel_path,
        projects_path,
        cpi_path,
        egresos_path,
        cambios_path=cambios_path,
        ajustes_path=ajustes_path,
        transcripcion_2022_path=transcripcion_path,
        cruce_manual_path=manual_path,
    )
    assert failed[0]["estado"] == "fallida"
    with pytest.raises(ValueError, match="cifras no verificadas"):
        require_verified(failed)


def test_reconciliation_counts_are_recalculated_from_primary_references(
    tmp_path: Path,
) -> None:
    panel = pd.DataFrame(
        [
            ["A", 2020, "A", "A", 2030, 100.0, "BASE", "Oro", "Atacama"],
            ["S", 2020, "S", "S", 2030, 50.0, "BASE", "Oro", "Atacama"],
            ["S", 2021, "S", "S", 2030, 50.0, "BASE", "Oro", "Atacama"],
            ["N", 2021, "N", "N", 2030, 200.0, "BASE", "Oro", "Atacama"],
        ],
        columns=[
            "project_id",
            "edicion",
            "proyecto_raw",
            "nombre_norm",
            "puesta_en_marcha",
            "inversion_mmusd",
            "condicion",
            "sector",
            "region",
        ],
    )
    projects = pd.DataFrame(
        [
            ["A", "entro_en_operacion"],
            ["S", "vigente"],
            ["N", "vigente"],
        ],
        columns=["project_id", "desenlace"],
    )
    panel_path = tmp_path / "panel.parquet"
    projects_path = tmp_path / "projects.parquet"
    cpi_path = tmp_path / "cpi.csv"
    egresos_path = tmp_path / "egresos.csv"
    cambios_path = tmp_path / "cambios.csv"
    ajustes_path = tmp_path / "ajustes.csv"
    panel.to_parquet(panel_path, index=False)
    projects.to_parquet(projects_path, index=False)
    pd.DataFrame(
        {"anio": [2020, 2021, 2025], "s1": [100, 100, 100], "s2": [100, 100, 100]}
    ).to_csv(cpi_path, index=False)
    pd.DataFrame(
        [
            [2020, 2021, "A", "operacion", "nombre"],
            [2021, 2025, "S", "no_declarado", "revision"],
            [2021, 2025, "N", "no_declarado", "revision"],
        ],
        columns=[
            "edicion_desde",
            "edicion_hasta",
            "proyecto_raw",
            "motivo",
            "fuente",
        ],
    ).to_csv(egresos_path, index=False)
    pd.DataFrame(
        [
            [2021, 2020, "entraron_operacion", 2, 100.0],
            [2021, 2020, "nuevos", 1, 200.0],
        ],
        columns=[
            "edicion",
            "par_desde",
            "categoria",
            "n_declarado",
            "capex_mmusd",
        ],
    ).to_csv(cambios_path, index=False)
    pd.DataFrame(
        [[2020, 2021, "entraron_operacion", 1, None]],
        columns=[
            "edicion_desde",
            "edicion_hasta",
            "categoria",
            "delta_n",
            "delta_capex",
        ],
    ).to_csv(ajustes_path, index=False)
    claims = [
        {"id": "n_conciliaciones_exactas", "valor": 1},
        {"id": "n_conciliaciones_con_ajuste", "valor": 1},
        {"id": "n_cifras_verificadas", "valor": 3},
    ]

    verified = verify_claims_register(
        claims,
        panel_path,
        projects_path,
        cpi_path,
        egresos_path,
        cambios_path=cambios_path,
        ajustes_path=ajustes_path,
    )

    assert {item["estado"] for item in verified} == {"verificada"}

    pd.DataFrame(
        [[2020, 2021, "entraron_operacion", 0, None]],
        columns=[
            "edicion_desde",
            "edicion_hasta",
            "categoria",
            "delta_n",
            "delta_capex",
        ],
    ).to_csv(ajustes_path, index=False)
    with pytest.raises(ValueError, match="2020.*2021.*entraron_operacion"):
        verify_claims_register(
            claims,
            panel_path,
            projects_path,
            cpi_path,
            egresos_path,
            cambios_path=cambios_path,
            ajustes_path=ajustes_path,
        )


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
    cif = build_cif_table(indicators.proyectos_deriva, bootstrap_replicates=1_000)
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
            for scenario in ("principal", "sin_2022", "sin_estatales")
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
    assert values["auc_corte_unico_completo"] == 0.7
    assert values["n_escenarios_potencial_positivo_p05"] == 3
    assert values["pct_regla_fecha_discrepante"] == 50


def test_altered_descriptive_claim_is_detected(tmp_path: Path) -> None:
    panel = pd.DataFrame(
        {
            "project_id": ["P1", "P1"],
            "edicion": [2020, 2021],
            "nombre_norm": ["P1", "P1"],
            "puesta_en_marcha": [2022, 2023],
            "inversion_mmusd": [100.0, 110.0],
            "condicion": ["BASE", "BASE"],
            "sector": ["GRAN_MINERIA_CU", "GRAN_MINERIA_CU"],
            "region": ["Antofagasta", "Antofagasta"],
        }
    )
    projects = pd.DataFrame(
        {
            "project_id": ["P1"],
            "primera_edicion": [2020],
            "ultima_edicion": [2021],
            "n_ediciones": [2],
            "pem_primera": [2022],
            "capex_primero": [100.0],
            "desenlace": ["vigente"],
            "fuente_desenlace": ["vigente"],
            "reingreso": [False],
        }
    )
    cpi_path = tmp_path / "cpi.csv"
    pd.DataFrame(
        {"anio": [2020, 2021, 2025], "s1": [100, 100, 100], "s2": [100, 100, 100]}
    ).to_csv(cpi_path, index=False)
    panel_path = tmp_path / "panel.parquet"
    projects_path = tmp_path / "projects.parquet"
    panel.to_parquet(panel_path, index=False)
    projects.to_parquet(projects_path, index=False)
    indicators = build_indicator_tables(panel, projects, cpi_path)
    descriptive, _ = build_descriptive_tables(
        panel,
        projects,
        indicators.transiciones,
        indicators.proyectos_deriva,
        pd.DataFrame(),
        cpi_path,
    )
    claims = build_claims_register(indicators, descriptivas=descriptive)
    altered = [dict(item) for item in claims]
    target = next(item for item in altered if item["id"] == "pct_atraso_base")
    target["valor"] = float(target["valor"]) + 1
    verified = verify_claims_register(
        altered, panel_path, projects_path, cpi_path
    )
    state = next(item["estado"] for item in verified if item["id"] == "pct_atraso_base")
    assert state == "fallida"


def test_altered_year_four_dashboard_claim_is_detected(tmp_path: Path) -> None:
    rows = []
    projects = []
    for index in range(10):
        project_id = f"P{index}"
        for edition, pem in ((2018, 2020), (2022, 2020 + index)):
            rows.append(
                {"project_id": project_id, "edicion": edition, "nombre_norm": project_id, "puesta_en_marcha": pem, "inversion_mmusd": 100, "condicion": "BASE", "sector": "Oro", "region": "Atacama"}
            )
        projects.append({"project_id": project_id, "desenlace": "vigente"})
    panel_path = tmp_path / "panel.parquet"
    projects_path = tmp_path / "projects.parquet"
    cpi_path = tmp_path / "cpi.csv"
    pd.DataFrame(rows).to_parquet(panel_path, index=False)
    pd.DataFrame(projects).to_parquet(projects_path, index=False)
    pd.DataFrame({"anio": [2018, 2022, 2025], "s1": [100, 100, 100], "s2": [100, 100, 100]}).to_csv(cpi_path, index=False)
    claims = [
        {"id": "mediana_deriva_anio4", "valor": 99.0},
        {"id": "p90_deriva_anio4", "valor": 8.1},
        {"id": "n_proyectos_anio4", "valor": 10},
        {"id": "n_cifras_verificadas", "valor": 4},
    ]
    verified = verify_claims_register(claims, panel_path, projects_path, cpi_path)
    states = {item["id"]: item["estado"] for item in verified}
    assert states["mediana_deriva_anio4"] == "fallida"
    assert states["p90_deriva_anio4"] == "verificada"
    assert states["n_proyectos_anio4"] == "verificada"
    assert states["n_cifras_verificadas"] == "fallida"
