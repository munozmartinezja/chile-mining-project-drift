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
    assert "## Robustez" in text
    assert "## Validación temporal" in text
    assert "## Regla de fecha contra desenlace declarado" in text


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


def test_report_includes_outcome_reconciliation_and_sources(tmp_path: Path) -> None:
    report = build_report(
        [],
        [],
        [
            {
                "desenlace": "censurado",
                "fuente_desenlace": "revision",
                "reingreso": False,
            }
        ],
        {"sector": [], "tipo_proyecto": [], "etapa": []},
        outcome_reconciliation=[
            {
                "edicion_desde": 2020,
                "edicion_hasta": 2021,
                "categoria": "metodologia",
                "decl_n": 1,
                "obs_n": 1,
                "decl_capex": 100,
                "obs_capex": 100,
                "delta_n": 0,
                "delta_capex": 0,
                "compara_capex": True,
                "estado": "ok",
            }
        ],
    )

    markdown_path, _ = write_report(report, tmp_path)
    text = markdown_path.read_text(encoding="utf-8")
    assert "## Conciliación de desenlaces" in text
    assert "### Fuentes de desenlace" in text
    assert report["desenlaces"]["fuentes"] == {"revision": 1}
