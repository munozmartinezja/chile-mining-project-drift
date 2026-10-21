from cpdrift.report import DECISIONES, build_report


def test_report_has_required_header_and_sections() -> None:
    report = build_report([], [], [], {"sector": [], "tipo_proyecto": [], "etapa": []})
    assert report["encabezado"] == "EXPLORATORIO — no publicable"
    assert set(report) >= {"ediciones", "identidad", "desenlaces", "catalogos", "decisiones_tomadas"}
    assert report["decisiones_tomadas"] == DECISIONES
