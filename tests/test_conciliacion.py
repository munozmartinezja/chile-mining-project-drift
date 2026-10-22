from pathlib import Path

import pandas as pd
import pytest

from cpdrift.conciliacion import (
    ReconciliationReferences,
    build_reconciliation,
    load_reconciliation_references,
    require_reconciled,
    verify_citations,
)


def panel_row(
    project_id: str,
    edition: int,
    name: str,
    capex: float,
) -> dict[str, object]:
    return {
        "project_id": project_id,
        "edicion": edition,
        "proyecto_raw": name,
        "nombre_norm": name.upper(),
        "inversion_mmusd": capex,
    }


def references(
    *,
    declared: list[tuple[str, int, float]] | None = None,
    exits: list[tuple[str, str]] | None = None,
    adjustments: list[tuple[str, int, float | None]] | None = None,
) -> ReconciliationReferences:
    declared = declared or [
        ("entraron_operacion", 1, 100.0),
        ("reestructuracion", 1, 200.0),
        ("nuevos", 1, 50.0),
    ]
    exits = exits or [("Operado", "operacion"), ("Retirado", "reestructuracion")]
    adjustments = adjustments or []
    return ReconciliationReferences(
        cambios=pd.DataFrame(
            [
                {
                    "edicion": 2021,
                    "par_desde": 2020,
                    "categoria": category,
                    "n_declarado": count,
                    "capex_mmusd": capex,
                    "pagina": "1",
                    "frase": "",
                }
                for category, count, capex in declared
            ]
        ),
        egresos=pd.DataFrame(
            [
                {
                    "edicion_desde": 2020,
                    "edicion_hasta": 2021,
                    "proyecto_raw": name,
                    "motivo": reason,
                    "fuente": "revision",
                    "edicion_fuente": 2021,
                    "pagina": "",
                    "frase": "",
                    "nota": "",
                }
                for name, reason in exits
            ]
        ),
        ajustes=pd.DataFrame(
            [
                {
                    "edicion_desde": 2020,
                    "edicion_hasta": 2021,
                    "categoria": category,
                    "delta_n": count,
                    "delta_capex": capex,
                    "nota": "",
                }
                for category, count, capex in adjustments
            ]
        ),
    )


@pytest.fixture
def panel() -> pd.DataFrame:
    return pd.DataFrame(
        [
            panel_row("A", 2020, "Operado", 100),
            panel_row("B", 2020, "Retirado", 200),
            panel_row("S", 2020, "Permanece", 300),
            panel_row("S", 2021, "Permanece", 300),
            panel_row("N", 2021, "Nuevo", 50),
        ]
    )


def test_reconciliation_gate_accepts_matching_counts_and_capex(panel: pd.DataFrame) -> None:
    table = build_reconciliation(panel, references())
    require_reconciled(table)
    assert set(table["estado"]) == {"ok"}


def test_reconciliation_fails_for_exit_without_reference(panel: pd.DataFrame) -> None:
    refs = references(exits=[("Operado", "operacion")])

    with pytest.raises(ValueError, match=r"salidas.*Retirado"):
        build_reconciliation(panel, refs)


def test_reconciliation_fails_for_count_difference(panel: pd.DataFrame) -> None:
    refs = references(
        declared=[
            ("entraron_operacion", 2, 100),
            ("reestructuracion", 1, 200),
            ("nuevos", 1, 50),
        ]
    )
    table = build_reconciliation(panel, refs)

    with pytest.raises(ValueError, match=r"decl_n[\s\S]*entraron_operacion"):
        require_reconciled(table)


def test_reconciliation_capex_tolerance_is_strictly_below_half(
    panel: pd.DataFrame,
) -> None:
    passing = references(
        declared=[
            ("entraron_operacion", 1, 100.49),
            ("reestructuracion", 1, 200),
            ("nuevos", 1, 50),
        ]
    )
    failing = references(
        declared=[
            ("entraron_operacion", 1, 100.5),
            ("reestructuracion", 1, 200),
            ("nuevos", 1, 50),
        ]
    )

    require_reconciled(build_reconciliation(panel, passing))
    with pytest.raises(ValueError, match="entraron_operacion"):
        require_reconciled(build_reconciliation(panel, failing))


def test_blank_capex_adjustment_skips_only_capex_comparison(panel: pd.DataFrame) -> None:
    refs = references(
        declared=[
            ("entraron_operacion", 2, 999),
            ("reestructuracion", 1, 200),
            ("nuevos", 1, 50),
        ],
        adjustments=[("entraron_operacion", 1, None)],
    )

    table = build_reconciliation(panel, refs)
    row = table.loc[table["categoria"].eq("entraron_operacion")].iloc[0]
    assert not bool(row["compara_capex"])
    require_reconciled(table)


def test_observed_category_without_declaration_is_compared_to_zero(
    panel: pd.DataFrame,
) -> None:
    refs = references(
        declared=[("entraron_operacion", 1, 100), ("nuevos", 1, 50)]
    )
    table = build_reconciliation(panel, refs)
    row = table.loc[table["categoria"].eq("reestructuracion")].iloc[0]

    assert row["decl_n"] == 0
    assert row["obs_n"] == 1
    assert row["estado"] == "falla"


def test_citation_is_found_in_2025_workbook_sheet(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    workbook = raw_dir / "anexo_c_2025.xlsx"
    pd.DataFrame([["Proyectos que salen por alcance."]]).to_excel(
        workbook,
        sheet_name="Tabla 2 -3",
        header=False,
        index=False,
    )
    refs = ReconciliationReferences(
        cambios=pd.DataFrame(
            [
                {
                    "edicion": 2025,
                    "par_desde": 2024,
                    "categoria": "metodologia",
                    "n_declarado": 1,
                    "capex_mmusd": 30,
                    "pagina": "Tabla 2 -3",
                    "frase": "proyectos que salen por alcance.",
                }
            ]
        ),
        egresos=pd.DataFrame(
            columns=[
                "edicion_desde",
                "edicion_hasta",
                "proyecto_raw",
                "motivo",
                "fuente",
                "edicion_fuente",
                "pagina",
                "frase",
                "nota",
            ]
        ),
        ajustes=pd.DataFrame(
            columns=[
                "edicion_desde",
                "edicion_hasta",
                "categoria",
                "delta_n",
                "delta_capex",
                "nota",
            ]
        ),
    )

    verify_citations(refs, raw_dir)


def test_reference_loader_rejects_invalid_domain(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    pd.DataFrame(
        [[2021, 2020, "desconocida", 1, 1, 1, "texto"]],
        columns=[
            "edicion",
            "par_desde",
            "categoria",
            "n_declarado",
            "capex_mmusd",
            "pagina",
            "frase",
        ],
    ).to_csv(reference_dir / "cambios_cartera_declarados.csv", index=False)
    pd.DataFrame(
        columns=[
            "edicion_desde",
            "edicion_hasta",
            "proyecto_raw",
            "motivo",
            "fuente",
            "edicion_fuente",
            "pagina",
            "frase",
            "nota",
        ]
    ).to_csv(reference_dir / "egresos_declarados.csv", index=False)
    pd.DataFrame(
        columns=[
            "edicion_desde",
            "edicion_hasta",
            "categoria",
            "delta_n",
            "delta_capex",
            "nota",
        ]
    ).to_csv(reference_dir / "ajustes_conciliacion.csv", index=False)

    with pytest.raises(ValueError, match=r"categoria.*desconocida"):
        load_reconciliation_references(reference_dir)
