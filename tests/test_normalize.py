import pandas as pd
import pytest

from cpdrift.normalize import (
    _catalog_key,
    apply_manual_regions,
    normalize_projects,
)


def source_row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "edicion": 2024,
        "puesta_en_marcha": 2028,
        "puesta_en_marcha_raw": pd.NA,
        "proyecto_raw": "AMP. MG. LOS PELAMBRES",
        "operador_raw": "Codelco Div. El Teniente S.A.",
        "sector": "Cobre",
        "region": "Antofagast a",
        "tipo_proyecto": "Brownfield",
        "condicion": "Posible",
        "etapa": pd.NA,
        "estado_permisos": pd.NA,
        "inversion_mmusd": 100.0,
        "rotulo_inversion": "Inversión 2024-2033 (MMUS$)",
        "pagina_fuente": 10,
    }
    row.update(changes)
    return row


@pytest.mark.parametrize("value", ["POSIBLE", "Posible"])
def test_normalization_catalogs_names_and_operator(value: str) -> None:
    frame, unknowns = normalize_projects(pd.DataFrame([source_row(condicion=value)]))
    row = frame.iloc[0]
    assert row["region"] == "Antofagasta"
    assert row["condicion"] == "POSIBLE"
    assert row["nombre_norm"] == "AMPLIACION MARGINAL LOS PELAMBRES"
    assert row["operador_norm"] == "CODELCO"
    assert unknowns == {"sector": [], "tipo_proyecto": [], "etapa": []}


def test_unknown_condition_aborts() -> None:
    with pytest.raises(ValueError, match="condicion desconocida.*MISTERIOSA"):
        normalize_projects(pd.DataFrame([source_row(condicion="Misteriosa")]))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Estatal - Cu", "ESTATAL_CU"),
        ("Estatal – Cu/ Plantas Met.", "ESTATAL_CU"),
        ("Gran Min - Cu", "GRAN_MINERIA_CU"),
        ("Gran Min – Cu", "GRAN_MINERIA_CU"),
        ("Gran Min. - Cu", "GRAN_MINERIA_CU"),
        ("Med Min - Cu", "MEDIANA_MINERIA_CU"),
        ("Med Min. - Cu", "MEDIANA_MINERIA_CU"),
        ("Med. Min. - Cu", "MEDIANA_MINERIA_CU"),
        ("Min Industrial", "MINERIA_INDUSTRIAL"),
        ("Min. Ind.", "MINERIA_INDUSTRIAL"),
        ("Min. Industrial", "MINERIA_INDUSTRIAL"),
        ("Plantas MeT.", "PLANTAS_METALURGICAS"),
        ("Plantas Met.", "PLANTAS_METALURGICAS"),
        ("Plantas Metalurgicas", "PLANTAS_METALURGICAS"),
        ("Plantas Metalúrg.", "PLANTAS_METALURGICAS"),
        ("Plantas Metalúrgica s", "PLANTAS_METALURGICAS"),
        ("Plantas Metalúrgicas", "PLANTAS_METALURGICAS"),
    ],
)
def test_reported_sector_labels_map_to_catalog(value: str, expected: str) -> None:
    frame, unknowns = normalize_projects(pd.DataFrame([source_row(sector=value)]))

    assert frame.loc[0, "sector"] == expected
    assert unknowns["sector"] == []


def test_reported_project_type_maps_to_catalog() -> None:
    frame, unknowns = normalize_projects(
        pd.DataFrame([source_row(tipo_proyecto="Reposición/ Nuevo")])
    )

    assert frame.loc[0, "tipo_proyecto"] == "REPOSICION_NUEVO"
    assert unknowns["tipo_proyecto"] == []


def test_catalog_key_preserves_word_spaces_and_normalizes_separators() -> None:
    assert (
        _catalog_key(" Gran  Min. – Cu / Plantas Metalúrgica s ")
        == "gran min-cu/plantas metalurgicas"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [("Ejecución/ Factibilidad", "EJECUCION_FACTIBILIDAD")],
)
def test_reported_stage_label_maps_to_catalog(value: str, expected: str) -> None:
    frame, unknowns = normalize_projects(pd.DataFrame([source_row(etapa=value)]))

    assert frame.loc[0, "etapa"] == expected
    assert unknowns["etapa"] == []


def _manual_regions(path, rows) -> None:
    pd.DataFrame(rows, columns=["edicion", "proyecto_raw", "region", "nota"]).to_csv(
        path, index=False
    )


def test_manual_region_anchor_fills_all_missing_appearances(tmp_path) -> None:
    path = tmp_path / "regiones.csv"
    _manual_regions(path, [[2014, "Proyecto A", "O'Higgins", "nota"]])
    panel = pd.DataFrame(
        {
            "edicion": [2014, 2015],
            "proyecto_raw": ["Proyecto A", "Proyecto A actualizado"],
            "project_id": [1, 1],
            "nombre_norm": ["PROYECTO A", "PROYECTO A ACTUALIZADO"],
            "region": [pd.NA, pd.NA],
            "region_imputada": [False, False],
        }
    )
    result = apply_manual_regions(panel, path)
    assert result["region"].tolist() == ["O'Higgins", "O'Higgins"]
    assert result["region_imputada"].tolist() == [True, True]


def test_manual_region_requires_existing_unique_anchor(tmp_path) -> None:
    path = tmp_path / "regiones.csv"
    _manual_regions(path, [[2014, "Ausente", "Atacama", "nota"]])
    panel = pd.DataFrame(
        {
            "edicion": [2014],
            "proyecto_raw": ["Otro"],
            "project_id": [1],
            "nombre_norm": ["OTRO"],
            "region": ["Atacama"],
            "region_imputada": [False],
        }
    )
    with pytest.raises(ValueError, match="ancla.*Ausente.*0"):
        apply_manual_regions(panel, path)

    duplicate = pd.concat([panel.assign(proyecto_raw="Ausente")] * 2, ignore_index=True)
    with pytest.raises(ValueError, match="ancla.*Ausente.*2"):
        apply_manual_regions(duplicate, path)


def test_manual_regions_reject_remaining_nonaggregate_missing_region(tmp_path) -> None:
    path = tmp_path / "regiones.csv"
    _manual_regions(path, [])
    panel = pd.DataFrame(
        {
            "edicion": [2014, 2014],
            "proyecto_raw": ["Proyecto", "Otros"],
            "project_id": [1, 2],
            "nombre_norm": ["PROYECTO", "OTROS PROYECTOS"],
            "region": [pd.NA, pd.NA],
            "region_imputada": [False, False],
        }
    )
    with pytest.raises(ValueError, match="proyectos sin región.*1"):
        apply_manual_regions(panel, path)


def test_manual_regions_allow_project_with_a_known_region(tmp_path) -> None:
    path = tmp_path / "regiones.csv"
    _manual_regions(path, [])
    panel = pd.DataFrame(
        {
            "edicion": [2023, 2024],
            "proyecto_raw": ["Proyecto", "Proyecto"],
            "project_id": [1, 1],
            "nombre_norm": ["PROYECTO", "PROYECTO"],
            "region": ["Valparaíso", pd.NA],
            "region_imputada": [False, False],
        }
    )
    result = apply_manual_regions(panel, path)
    assert result.loc[0, "region"] == "Valparaíso"
    assert pd.isna(result.loc[1, "region"])
