from pathlib import Path

import pandas as pd

from cpdrift.export_powerbi import export_powerbi


def test_export_is_anonymous_and_uses_bom_and_dot_decimal(tmp_path: Path) -> None:
    panel = pd.DataFrame(
        {"project_id": [1], "proyecto_raw": ["Nombre reservado"]}
    )
    projects = pd.DataFrame(
        {
            "project_id": [1],
            "id_publico": ["C3-001"],
            "sector": ["ORO"],
            "region": ["NORTE"],
            "condicion_inicial": ["BASE"],
            "primera_edicion": [2020],
            "ultima_edicion": [2025],
            "n_ediciones": [2],
            "desenlace": ["vigente"],
            "fuente_desenlace": ["vigente"],
            "reingreso": [False],
            "capex_inicial_real": [12.5],
        }
    )
    tables = {
        "trayectorias": pd.DataFrame(
            {"id_publico": ["C3-001"], "edicion": [2020], "capex_real": [12.5]}
        ),
        "desc_cartera_edicion": pd.DataFrame(
            {"edicion": [2020], "capex_nominal_total": [12.5]}
        ),
    }
    paths = export_powerbi(
        tmp_path,
        panel=panel,
        projects=projects,
        transitions=pd.DataFrame(),
        descriptive=tables,
        cif=pd.DataFrame({"cohorte": pd.Series(dtype=str)}),
        reconciliation=pd.DataFrame({"categoria": pd.Series(dtype=str)}),
        validation_cuts=pd.DataFrame({"corte": pd.Series(dtype=int)}),
        validation_aggregate=pd.DataFrame({"modelo": pd.Series(dtype=str)}),
        aggregate_calibration=pd.DataFrame({"quintil": pd.Series(dtype=int)}),
        sensitivity=pd.DataFrame({"escenario": pd.Series(dtype=str)}),
        date_rule_matrix=pd.DataFrame({"declarado": pd.Series(dtype=str)}),
        copper_prices=pd.DataFrame({"edicion": pd.Series(dtype=int)}),
        claims=[],
    )
    for path in paths:
        assert path.read_bytes().startswith(b"\xef\xbb\xbf")
        exported = pd.read_csv(path)
        forbidden_columns = [
            column
            for column in exported.columns
            if column == "proyecto_raw"
            or column.startswith(("operador_", "nombre_"))
            or column == "project_id"
        ]
        assert forbidden_columns == []
        assert "Nombre reservado" not in exported.astype(str).to_numpy()
    portfolio_text = (tmp_path / "desc_cartera_edicion.csv").read_text(
        encoding="utf-8-sig"
    )
    assert "12.5" in portfolio_text
