"""Entry point: uv run python -m cpdrift"""

from __future__ import annotations

from pathlib import Path

from .claims import build_claims_register
from .conciliacion import (
    build_date_rule_check,
    build_reconciliation,
    load_reconciliation_references,
    require_reconciled,
    verify_citations,
    write_date_rule_check,
    write_reconciliation,
)
from .declared import extract_declared_all
from .descriptivo import (
    build_descriptive_tables,
    build_fan_table,
    build_reconciliation_matrix,
    build_sensitivity_potential,
    build_ten_table,
    write_descriptive_tables,
    write_visual_tables,
)
from .export_powerbi import export_powerbi
from .extract import apply_extraction_corrections, extract_all
from .indicators import build_indicator_tables, write_indicator_tables
from .match import assign_project_ids
from .normalize import apply_manual_regions, impute_regions, normalize_projects
from .panel import write_outputs
from .precios import build_copper_prices, write_copper_prices
from .regression import (
    fit_regressions_from_parquet,
    prepare_extended_regression_data,
    write_regressions,
)
from .report import build_report, write_report
from .sensibilidad import build_sensitivity, write_sensitivity
from .survival import build_cif_from_parquet, write_cif
from .validacion import (
    build_successive_validation,
    build_temporal_validation,
    write_successive_validation,
    write_temporal_validation,
)
from .verify import require_verified, verify_claims_register, write_claims_register

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
MANUAL_PATH = Path("data/reference/cruce_manual.csv")
REVISION_PATH = INTERIM_DIR / "revision_cruce.csv"
CORRECTIONS_PATH = Path("data/reference/correcciones_extraccion.csv")
CPI_PATH = Path("data/reference/cpi_us_semestral.csv")
REFERENCE_DIR = Path("data/reference")
COPPER_PRICE_PATH = RAW_DIR / "cmo-precios-mensual.xlsx"
MANUAL_REGIONS_PATH = REFERENCE_DIR / "regiones_manual.csv"


def main() -> None:
    print("Extrayendo filas…")
    extracted, _ = extract_all(RAW_DIR)
    extracted = apply_extraction_corrections(extracted, CORRECTIONS_PATH)
    print(f"  {len(extracted)} filas en {extracted['edicion'].nunique()} ediciones")

    print("Conciliando totales declarados…")
    reconciliation = extract_declared_all(RAW_DIR, extracted)

    print("Normalizando…")
    normalized, unknowns = normalize_projects(extracted)

    print("Asignando identidad…")
    identified, identity_metrics, _applied = assign_project_ids(
        normalized, MANUAL_PATH, REVISION_PATH
    )

    print("Imputando regiones…")
    panel = impute_regions(identified)
    panel = apply_manual_regions(panel, MANUAL_REGIONS_PATH)

    threshold_by_edition = {
        item["edicion"]: item["umbral_edicion"] for item in reconciliation
    }
    panel["umbral_edicion"] = panel["edicion"].map(threshold_by_edition)

    print("Conciliando desenlaces…")
    outcome_references = load_reconciliation_references(REFERENCE_DIR)
    verify_citations(outcome_references, RAW_DIR)
    outcome_reconciliation = build_reconciliation(panel, outcome_references)
    require_reconciled(outcome_reconciliation)
    write_reconciliation(outcome_reconciliation, INTERIM_DIR)

    print("Escribiendo parquets…")
    panel_df, projects_df = write_outputs(
        panel, INTERIM_DIR, outcome_references.egresos
    )

    print("Calculando indicadores…")
    indicators = build_indicator_tables(
        panel_df, projects_df, CPI_PATH, outcome_references.egresos
    )
    write_indicator_tables(indicators, INTERIM_DIR)
    descriptive, projects_df = build_descriptive_tables(
        panel_df,
        projects_df,
        indicators.transiciones,
        indicators.proyectos_deriva,
        outcome_reconciliation,
        CPI_PATH,
    )
    projects_df.to_parquet(INTERIM_DIR / "proyectos.parquet", index=False)
    write_descriptive_tables(descriptive, INTERIM_DIR)

    print("Estimando modelos asociativos…")
    cif = build_cif_from_parquet(INTERIM_DIR / "proyectos_deriva.parquet")
    write_cif(cif, INTERIM_DIR)
    copper_prices = build_copper_prices(
        COPPER_PRICE_PATH, CPI_PATH, set(panel_df["edicion"])
    )
    copper_prices_path = write_copper_prices(copper_prices, INTERIM_DIR)
    regressions = fit_regressions_from_parquet(
        INTERIM_DIR / "transiciones.parquet",
        INTERIM_DIR / "panel.parquet",
        copper_prices_path,
    )
    write_regressions(regressions, INTERIM_DIR)

    print("Evaluando robustez y validación temporal…")
    extended_data = prepare_extended_regression_data(
        indicators.transiciones, panel_df, copper_prices
    )
    validation = build_temporal_validation(extended_data)
    write_temporal_validation(validation, INTERIM_DIR)
    successive_validation = build_successive_validation(extended_data)
    write_successive_validation(successive_validation, INTERIM_DIR)
    sensitivity_projects = indicators.proyectos_deriva.merge(
        projects_df[["project_id", "fuente_desenlace"]],
        on="project_id",
        how="left",
        validate="one_to_one",
    )
    sensitivity = build_sensitivity(extended_data, sensitivity_projects, panel_df)
    write_sensitivity(sensitivity, INTERIM_DIR)
    date_rule = build_date_rule_check(panel_df, outcome_references.egresos)
    write_date_rule_check(date_rule, INTERIM_DIR)
    visual_tables = {
        "viz_abanico": build_fan_table(descriptive["trayectorias"]),
        "viz_de_cada_10": build_ten_table(cif),
        "viz_sensibilidad_potencial": build_sensitivity_potential(sensitivity),
        "viz_conciliacion_matriz": build_reconciliation_matrix(
            outcome_reconciliation
        ),
    }
    write_visual_tables(visual_tables, INTERIM_DIR)

    print("Verificando cifras…")
    claims = build_claims_register(
        indicators,
        cif=cif,
        ols=regressions.ols,
        logit=regressions.logit,
        projects=projects_df,
        egresos=outcome_references.egresos,
        conciliacion=outcome_reconciliation,
        ols_ampliado=regressions.ols_ampliado,
        validacion=validation.metricas,
        validacion_cortes=successive_validation.cortes,
        validacion_agregada=successive_validation.agregada,
        validacion_bootstrap=successive_validation.bootstrap,
        descriptivas=descriptive,
        visuales=visual_tables,
        sensibilidad=sensitivity,
        regla_fecha=date_rule.detalle,
    )
    verified_claims = verify_claims_register(
        claims,
        INTERIM_DIR / "panel.parquet",
        INTERIM_DIR / "proyectos.parquet",
        CPI_PATH,
        REFERENCE_DIR / "egresos_declarados.csv",
        COPPER_PRICE_PATH,
    )
    write_claims_register(verified_claims, INTERIM_DIR / "claims.json")
    require_verified(verified_claims)

    print("Generando reporte…")
    report = build_report(
        editions=reconciliation,
        identity=identity_metrics,
        projects=projects_df.to_dict("records"),
        unknowns=unknowns,
        indicators=verified_claims,
        models={
            "cif": cif.to_dict("records"),
            "ols": regressions.ols.to_dict("records"),
            "logit": regressions.logit.to_dict("records"),
            "ols_ampliado": regressions.ols_ampliado.to_dict("records"),
        },
        outcome_reconciliation=outcome_reconciliation.to_dict("records"),
        robustness=sensitivity.to_dict("records"),
        temporal_validation=successive_validation.cortes.to_dict("records"),
        aggregate_validation=successive_validation.agregada.to_dict("records"),
        calibration=successive_validation.calibracion.to_dict("records"),
        single_cut_validation=validation.metricas.to_dict("records"),
        validation_intervals=successive_validation.bootstrap,
        descriptive={
            name: table.to_dict("records") for name, table in descriptive.items()
        },
        date_rule=date_rule.matriz.to_dict("records"),
    )
    md_path, json_path = write_report(report, INTERIM_DIR)

    export_projects = projects_df.merge(
        indicators.proyectos_deriva[
            [
                "project_id",
                "sector",
                "region",
                "condicion_inicial",
                "capex_inicial_real",
            ]
        ],
        on="project_id",
        how="left",
        validate="one_to_one",
    )
    export_powerbi(
        Path("data/powerbi"),
        panel=panel_df,
        projects=export_projects,
        transitions=indicators.transiciones,
        descriptive=descriptive,
        cif=cif,
        reconciliation=outcome_reconciliation,
        validation_cuts=successive_validation.cortes,
        validation_aggregate=successive_validation.agregada,
        aggregate_calibration=successive_validation.calibracion,
        sensitivity=sensitivity,
        date_rule_matrix=date_rule.matriz,
        copper_prices=copper_prices,
        claims=verified_claims,
        visual_tables=visual_tables,
    )

    if REVISION_PATH.exists():
        with open(REVISION_PATH) as fh:
            revision_rows = sum(1 for _ in fh) - 1
    else:
        revision_rows = 0
    print(f"\nOK — panel: {len(panel_df)} filas | proyectos: {len(projects_df)}")
    print(f"  ambiguos en revision_cruce.csv: {revision_rows}")
    print(f"  reporte: {md_path}")
    print(f"  json: {json_path}")


if __name__ == "__main__":
    main()
