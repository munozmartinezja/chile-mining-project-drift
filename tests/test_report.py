import json
import math
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
        models={"cif": [], "ols": [], "logit": []},
    )
    assert report["encabezado"] == "EXPLORATORIO — no publicable"
    assert set(report) >= {
        "ediciones",
        "identidad",
        "desenlaces",
        "catalogos",
        "indicadores",
        "modelos",
        "decisiones_tomadas",
    }
    assert report["indicadores"] == indicators
    assert report["decisiones_tomadas"] == DECISIONES
    markdown_path, _ = write_report(report, tmp_path)
    assert "## Indicadores" in markdown_path.read_text(encoding="utf-8")
    text = markdown_path.read_text(encoding="utf-8")
    assert markdown_path.name == "diagnostico.md"
    assert (tmp_path / "diagnostico.json").exists()
    assert "## Modelos" in text
    assert "resultados asociativos" in text


def test_report_avoids_disallowed_interpretive_vocabulary() -> None:
    source = Path("src/cpdrift/report.py").read_text(encoding="utf-8").casefold()
    assert "causa" not in source
    assert "efecto de" not in source
    assert "provoca" not in source


def test_report_json_replaces_non_finite_model_values(tmp_path: Path) -> None:
    report = build_report(
        [],
        [],
        [],
        {"sector": [], "tipo_proyecto": [], "etapa": []},
        models={
            "cif": [
                {
                    "cohorte": "incidente",
                    "grupo": "TOTAL",
                    "t": 1,
                    "cif_op": 0.0,
                    "cif_sal": 0.0,
                    "ic_op_inf": math.nan,
                    "ic_op_sup": math.inf,
                    "ic_sal_inf": math.nan,
                    "ic_sal_sup": math.nan,
                }
            ],
            "ols": [],
            "logit": [],
        },
    )

    _, json_path = write_report(report, tmp_path)
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["modelos"]["cif"][0]["ic_op_inf"] is None
    assert saved["modelos"]["cif"][0]["ic_op_sup"] is None
