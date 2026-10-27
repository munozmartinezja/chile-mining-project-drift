from pathlib import Path

import pandas as pd
import pytest

from cpdrift.match import _score, assign_project_ids


def row(edition: int, name: str, operator: str = "ACME", region: str = "Atacama") -> dict[str, object]:
    return {
        "edicion": edition,
        "proyecto_raw": name,
        "nombre_norm": name,
        "operador_norm": operator,
        "region": region,
        "puesta_en_marcha": 2030,
        "inversion_mmusd": 100.0,
    }


def run_match(tmp_path: Path, rows: list[dict[str, object]], manual: str | None = None) -> pd.DataFrame:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manual_path = tmp_path / "manual.csv"
    text = manual or (
        "edicion,proyecto_raw,decision,destino_edicion,"
        "destino_proyecto_raw,nota\n"
    )
    source_lines = text.splitlines()
    upgraded = [source_lines[0] + ",evidencia"]
    for line in source_lines[1:]:
        action = line.split(",")[2]
        upgraded.append(line + (",nombre" if action == "mismo" else ",no_aplica"))
    manual_path.write_text("\n".join(upgraded) + "\n", encoding="utf-8")
    matched, _, _ = assign_project_ids(
        pd.DataFrame(rows), manual_path, tmp_path / "revision.csv"
    )
    return matched


def test_project_returning_after_exit_matches_all_history(tmp_path: Path) -> None:
    matched = run_match(tmp_path, [row(2016, "PROYECTO ALFA"), row(2019, "PROYECTO ALFA")])
    assert matched["project_id"].tolist() == ["P001", "P001"]


def test_manual_distinto_blocks_automatic_match(tmp_path: Path) -> None:
    manual = (
        "edicion,proyecto_raw,decision,destino_edicion,destino_proyecto_raw,nota\n"
        "2019,PROYECTO ALFA,distinto,2016,PROYECTO ALFA,homónimo\n"
    )
    matched = run_match(tmp_path, [row(2016, "PROYECTO ALFA"), row(2019, "PROYECTO ALFA")], manual)
    assert matched["project_id"].tolist() == ["P001", "P002"]


def test_manual_mismo_uses_anchor_when_prior_insertions_change_ids(tmp_path: Path) -> None:
    manual = (
        "edicion,proyecto_raw,decision,destino_edicion,destino_proyecto_raw,nota\n"
        "2019,PROYECTO BETA,mismo,2014,PROYECTO ALFA,continuidad confirmada\n"
    )
    base = run_match(
        tmp_path / "base",
        [row(2014, "PROYECTO ALFA"), row(2019, "PROYECTO BETA")],
        manual,
    )
    shifted = run_match(
        tmp_path / "shifted",
        [
            row(2014, "PROYECTO NUEVO"),
            row(2014, "PROYECTO ALFA"),
            row(2019, "PROYECTO BETA"),
        ],
        manual,
    )

    assert base["project_id"].tolist() == ["P001", "P001"]
    assert shifted["project_id"].tolist() == ["P001", "P002", "P002"]


def test_manual_nuevo_forces_distinct_project_id(tmp_path: Path) -> None:
    manual = (
        "edicion,proyecto_raw,decision,destino_edicion,destino_proyecto_raw,nota\n"
        "2019,PROYECTO ALFA,nuevo,,,identidad nueva\n"
    )

    matched = run_match(
        tmp_path,
        [row(2016, "PROYECTO ALFA"), row(2019, "PROYECTO ALFA")],
        manual,
    )

    assert matched["project_id"].tolist() == ["P001", "P002"]


def test_manual_missing_destination_raises(tmp_path: Path) -> None:
    manual = (
        "edicion,proyecto_raw,decision,destino_edicion,destino_proyecto_raw,nota\n"
        "2019,PROYECTO BETA,mismo,2014,PROYECTO AUSENTE,revisión\n"
    )

    with pytest.raises(ValueError, match=r"destino manual inexistente.*2014.*PROYECTO AUSENTE"):
        run_match(
            tmp_path,
            [row(2014, "PROYECTO ALFA"), row(2019, "PROYECTO BETA")],
            manual,
        )


def test_manual_future_destination_raises(tmp_path: Path) -> None:
    manual = (
        "edicion,proyecto_raw,decision,destino_edicion,destino_proyecto_raw,nota\n"
        "2016,PROYECTO ALFA,mismo,2019,PROYECTO BETA,revisión\n"
    )

    with pytest.raises(ValueError, match=r"destino manual debe ser anterior.*2016.*2019"):
        run_match(
            tmp_path,
            [row(2016, "PROYECTO ALFA"), row(2019, "PROYECTO BETA")],
            manual,
        )


def test_manual_missing_source_raises(tmp_path: Path) -> None:
    manual = (
        "edicion,proyecto_raw,decision,destino_edicion,destino_proyecto_raw,nota\n"
        "2019,PROYECTO AUSENTE,nuevo,,,revisión\n"
    )

    with pytest.raises(ValueError, match=r"origen manual inexistente.*2019.*PROYECTO AUSENTE"):
        run_match(tmp_path, [row(2019, "PROYECTO ALFA")], manual)


def test_old_manual_schema_is_migrated_to_empty_new_header(tmp_path: Path) -> None:
    manual_path = tmp_path / "manual.csv"
    manual_path.write_text(
        "edicion,proyecto_raw,decision,project_id_destino,nota\n",
        encoding="utf-8",
    )

    assign_project_ids(
        pd.DataFrame([row(2016, "PROYECTO ALFA")]),
        manual_path,
        tmp_path / "revision.csv",
    )

    assert manual_path.read_text(encoding="utf-8") == (
        "edicion,proyecto_raw,decision,destino_edicion,destino_proyecto_raw,nota,evidencia\n"
    )


def test_same_edition_rows_cannot_share_project_id(tmp_path: Path) -> None:
    matched = run_match(
        tmp_path,
        [row(2016, "PROYECTO ALFA"), row(2019, "PROYECTO ALFA"), row(2019, "PROYECTO ALFA")],
    )
    assert matched.loc[matched.edicion.eq(2019), "project_id"].nunique() == 2


def test_project_ids_are_deterministic(tmp_path: Path) -> None:
    rows = [row(2014, "BETA"), row(2014, "ALFA"), row(2015, "BETA")]
    first = run_match(tmp_path / "one", rows)
    second = run_match(tmp_path / "two", rows)
    assert first["project_id"].tolist() == second["project_id"].tolist() == ["P001", "P002", "P001"]


def test_score_ignores_region_when_either_or_both_are_null() -> None:
    source = row(2014, "PROYECTO ALFA", operator="UNO")
    candidate = row(2015, "PROYECTO ALFO", operator="DOS")
    source_null = pd.Series({**source, "region": pd.NA})
    source_known = pd.Series({**source, "region": "Atacama"})
    candidate_known = pd.Series({**candidate, "region": "Atacama"})
    candidate_null = pd.Series({**candidate, "region": pd.NA})
    both_null = pd.Series({**candidate, "region": pd.NA})

    without_region = _score(source_null, candidate_known)
    assert _score(source_known, candidate_null) == without_region
    assert _score(source_null, both_null) == without_region
    assert _score(source_known, candidate_known) == without_region + 3


def test_token_set_containment_without_token_sort_is_ambiguous(tmp_path: Path) -> None:
    matched = run_match(
        tmp_path,
        [
            {**row(2014, "SIERRA GORDA"), "puesta_en_marcha": 2014},
            {**row(2018, "SIERRA GORDA EXPANSION 220 KTPD"), "puesta_en_marcha": 2018},
        ],
    )

    assert matched["clasificacion_cruce"].tolist() == ["nuevo", "ambiguo"]
    assert matched["project_id"].tolist() == ["P001", "P002"]


def test_immediate_previous_edition_still_requires_token_sort_threshold(
    tmp_path: Path,
) -> None:
    matched = run_match(
        tmp_path,
        [
            {**row(2014, "SIERRA GORDA"), "puesta_en_marcha": 2014},
            {
                **row(2015, "SIERRA GORDA EXPANSION 220 KTPD"),
                "puesta_en_marcha": 2018,
            },
        ],
    )

    assert matched["project_id"].tolist() == ["P001", "P002"]


def test_nested_parenthetical_alias_matches_automatically(tmp_path: Path) -> None:
    matched = run_match(
        tmp_path,
        [
            row(
                2014,
                "Adecuaciones Ejecución Proyecto NNM "
                "(Plan de Desarrollo El Teniente (Nuevo Nivel Mina))",
            ),
            row(2018, "NUEVO NIVEL MINA"),
        ],
    )

    assert matched["clasificacion_cruce"].tolist() == ["nuevo", "automatico"]
    assert matched["project_id"].tolist() == ["P001", "P001"]


def test_higher_scoring_candidate_wins_when_scores_are_not_tied(tmp_path: Path) -> None:
    matched = run_match(
        tmp_path,
        [
            row(2014, "PROYECTO ALFA NORTE"),
            row(2014, "PROYECTO ALFA NORTA", operator="OTRA", region="Coquimbo"),
            row(2015, "PROYECTO ALFA NORTE"),
        ],
    )

    assert matched["clasificacion_cruce"].tolist() == ["nuevo", "nuevo", "automatico"]
    assert matched["project_id"].tolist() == ["P001", "P002", "P001"]


def test_revision_has_at_most_three_ranked_candidates_even_with_ties(tmp_path: Path) -> None:
    rows = [row(2014, f"PROYECTO COBRE NORTE {suffix}") for suffix in "ABCDE"]
    rows.append(row(2015, "PROYECTO COBRE NORTE"))
    run_match(tmp_path, rows)

    revision = pd.read_csv(tmp_path / "revision.csv", dtype=str, keep_default_na=False)

    assert len(revision) == 3
    assert revision["rango"].tolist() == ["1", "2", "3"]
    assert revision["decision"].tolist() == ["", "", ""]
    assert revision["candidato_edicion"].tolist() == ["2014", "2014", "2014"]
    assert revision["candidato_proyecto_raw"].str.startswith("PROYECTO COBRE NORTE").all()
    assert revision["puntaje"].astype(float).ge(60).all()


def test_dominga_returns_to_stable_id_after_low_sort_variant(tmp_path: Path) -> None:
    rows = [row(edition, "DOMINGA") for edition in range(2014, 2022)]
    rows.extend(
        [
            row(2023, "PROYECTO MINERO PORTUARIO DOMINGA"),
            row(2024, "DOMINGA"),
            row(2025, "DOMINGA"),
        ]
    )

    matched = run_match(tmp_path, rows)

    assert matched["project_id"].tolist() == ["P001"] * 8 + ["P002", "P001", "P001"]
    assert matched.loc[matched["edicion"].ge(2024), "clasificacion_cruce"].eq(
        "automatico"
    ).all()


def test_rajo_inca_returns_to_stable_id_after_low_sort_variant(tmp_path: Path) -> None:
    editions = [*range(2014, 2022), 2023]
    rows = [row(edition, "RAJO INCA") for edition in editions]
    rows.extend(
        [
            row(2024, "PROYECTO RAJO INCA (PRI)"),
            row(2025, "RAJO INCA"),
        ]
    )

    matched = run_match(tmp_path, rows)

    assert matched["project_id"].tolist() == ["P001"] * 9 + ["P002", "P001"]


def test_non_provisional_candidate_cuts_ambiguous_cascade(tmp_path: Path) -> None:
    matched = run_match(
        tmp_path,
        [
            row(2014, "MINA ALFA"),
            row(2020, "PROYECTO MINERO MINA ALFA"),
            row(2021, "MINA ALFA"),
        ],
    )

    assert matched["clasificacion_cruce"].tolist() == ["nuevo", "ambiguo", "automatico"]
    assert matched["project_id"].tolist() == ["P001", "P002", "P001"]


def test_low_token_sort_non_provisional_does_not_cut_provisional_chain(
    tmp_path: Path,
) -> None:
    chain_name = "SIERRA GORDA EXP 230 KTPD"
    rows = [{**row(2014, "SIERRA GORDA"), "puesta_en_marcha": 2014}]
    rows.extend(
        {**row(edition, chain_name), "puesta_en_marcha": 2018}
        for edition in [2017, 2018, 2019, 2020, 2021]
    )
    rows.extend([row(2023, "PROYECTO DISTINTO"), row(2024, chain_name)])

    matched = run_match(tmp_path, rows)
    chain = matched.loc[matched["proyecto_raw"].eq(chain_name)]

    assert chain["project_id"].nunique() == 1
    assert chain["project_id"].iloc[0] != matched.loc[0, "project_id"]


def test_score_tie_prefers_most_recent_candidate(tmp_path: Path) -> None:
    matched = run_match(
        tmp_path,
        [
            row(2014, "MINA ALFA NORTE A"),
            row(2014, "MINA ALFA NORTE B", operator="OTRA", region="Coquimbo"),
            row(2015, "MINA ALFA NORTE A"),
            row(2016, "MINA ALFA NORTE"),
        ],
    )

    assert matched["project_id"].tolist() == ["P001", "P002", "P001", "P001"]


def test_score_and_recency_tie_prefers_closest_commissioning(tmp_path: Path) -> None:
    matched = run_match(
        tmp_path,
        [
            {**row(2014, "MINA ALFA NORTE A"), "puesta_en_marcha": 2029},
            {
                **row(
                    2014,
                    "MINA ALFA NORTE B",
                    operator="OTRA",
                    region="Coquimbo",
                ),
                "puesta_en_marcha": 2040,
            },
            {**row(2015, "MINA ALFA NORTE"), "puesta_en_marcha": 2030},
        ],
    )

    assert matched["project_id"].tolist() == ["P001", "P002", "P001"]


def test_commissioning_tie_uses_each_candidates_latest_appearance(tmp_path: Path) -> None:
    matched = run_match(
        tmp_path,
        [
            {**row(2014, "MINA ALFA NORTE A"), "puesta_en_marcha": 2030},
            {
                **row(
                    2014,
                    "MINA ALFA NORTE B",
                    operator="OTRA",
                    region="Coquimbo",
                ),
                "puesta_en_marcha": 2040,
            },
            {**row(2015, "MINA ALFA NORTE A"), "puesta_en_marcha": 2050},
            {
                **row(
                    2015,
                    "MINA ALFA NORTE B",
                    operator="OTRA",
                    region="Coquimbo",
                ),
                "puesta_en_marcha": 2035,
            },
            {**row(2016, "MINA ALFA NORTE"), "puesta_en_marcha": 2030},
        ],
    )

    assert matched["project_id"].tolist() == ["P001", "P002", "P001", "P002", "P002"]


def test_numbered_phases_remain_distinct_and_stable(tmp_path: Path) -> None:
    rows = [
        row(edition, name)
        for edition in range(2016, 2022)
        for name in (
            "AMP MARGINAL LOS PELAMBRES FASE I",
            "AMP MARGINAL LOS PELAMBRES FASE II",
        )
    ]

    matched = run_match(tmp_path, rows)
    ids_by_name = matched.groupby("proyecto_raw")["project_id"]

    assert matched["project_id"].nunique() == 2
    assert ids_by_name.nunique().eq(1).all()
    assert matched.loc[matched.edicion.gt(2016), "clasificacion_cruce"].eq("automatico").all()


def test_unmarked_history_does_not_bridge_different_numbered_phases(
    tmp_path: Path,
) -> None:
    matched = run_match(
        tmp_path,
        [
            row(2014, "AMPLIACION MARGINAL LOS PELAMBRES"),
            row(2015, "AMPLIACION MARGINAL LOS PELAMBRES FASE I"),
            row(2016, "AMPLIACION MARGINAL LOS PELAMBRES FASE II"),
            row(2017, "AMPLIACION MARGINAL LOS PELAMBRES FASE I"),
            row(2017, "AMPLIACION MARGINAL LOS PELAMBRES FASE II"),
        ],
    )

    assert matched["project_id"].tolist() == ["P001", "P001", "P002", "P001", "P002"]


def test_capacity_numbers_are_not_phase_markers() -> None:
    source = pd.Series(row(2018, "SIERRA GORDA EXPANSION 220 KTPD"))
    candidate = pd.Series(row(2019, "SIERRA GORDA EXPANSION 230 KTPD"))

    assert _score(source, candidate) > 70
