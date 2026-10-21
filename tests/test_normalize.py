import pandas as pd
import pytest

from cpdrift.normalize import _catalog_key, normalize_projects


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
