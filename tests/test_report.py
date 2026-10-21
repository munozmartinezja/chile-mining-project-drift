from pathlib import Path

from cpdrift.report import DECISIONES, build_report, write_report


def test_report_has_required_header_and_sections(tmp_path: Path) -> None:
    indicators = [
        {
            "id": "n_proyectos",
            "descripcion": "Proyectos",
            "valor": 3,
            "unidad": "proyectos",
            "calculo": "Conteo.",
            "valor_recalculado": 3,
            "estado": "verificada",
        }
    ]
    report = build_report(
        [],
        [],
        [],
        {"sector": [], "tipo_proyecto": [], "etapa": []},
        indicators=indicators,
    )
    assert report["encabezado"] == "EXPLORATORIO — no publicable"
    assert set(report) >= {
        "ediciones",
        "identidad",
        "desenlaces",
        "catalogos",
        "indicadores",
        "decisiones_tomadas",
    }
    assert report["indicadores"] == indicators
    assert report["decisiones_tomadas"] == DECISIONES
    markdown_path, _ = write_report(report, tmp_path)
    assert "## Indicadores" in markdown_path.read_text(encoding="utf-8")
