import ast
from pathlib import Path

import pandas as pd
import pytest

from cpdrift.claims import build_claims_register
from cpdrift.indicators import build_indicator_tables
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
        ]
    )
    projects = pd.DataFrame(
        [
            {"project_id": "P1", "desenlace": "entro_en_operacion", "reingreso": False},
            {"project_id": "A1", "desenlace": "vigente", "reingreso": False},
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
    require_verified(verified)

    changed = [dict(item) for item in claims]
    changed[0]["valor"] += 1
    failed = verify_claims_register(changed, panel_path, projects_path, cpi_path)
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
