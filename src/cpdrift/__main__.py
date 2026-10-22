"""Entry point: uv run python -m cpdrift"""

from __future__ import annotations

from pathlib import Path

from .claims import build_claims_register
from .declared import extract_declared_all
from .extract import apply_extraction_corrections, extract_all
from .indicators import build_indicator_tables, write_indicator_tables
from .match import assign_project_ids
from .normalize import impute_regions, normalize_projects
from .panel import write_outputs
from .report import build_report, write_report
from .verify import require_verified, verify_claims_register, write_claims_register

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
MANUAL_PATH = Path("data/reference/cruce_manual.csv")
REVISION_PATH = INTERIM_DIR / "revision_cruce.csv"
CORRECTIONS_PATH = Path("data/reference/correcciones_extraccion.csv")
CPI_PATH = Path("data/reference/cpi_us_semestral.csv")


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

    threshold_by_edition = {
        item["edicion"]: item["umbral_edicion"] for item in reconciliation
    }
    panel["umbral_edicion"] = panel["edicion"].map(threshold_by_edition)

    print("Escribiendo parquets…")
    panel_df, projects_df = write_outputs(panel, INTERIM_DIR)

    print("Calculando indicadores…")
    indicators = build_indicator_tables(panel_df, projects_df, CPI_PATH)
    write_indicator_tables(indicators, INTERIM_DIR)

    print("Verificando cifras…")
    claims = build_claims_register(indicators)
    verified_claims = verify_claims_register(
        claims,
        INTERIM_DIR / "panel.parquet",
        INTERIM_DIR / "proyectos.parquet",
        CPI_PATH,
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
    )
    md_path, json_path = write_report(report, INTERIM_DIR)

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
