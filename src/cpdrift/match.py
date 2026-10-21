"""Deterministic longitudinal identity assignment."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from .normalize import normalize_name

MANUAL_COLUMNS = [
    "edicion",
    "proyecto_raw",
    "decision",
    "destino_edicion",
    "destino_proyecto_raw",
    "nota",
]
MANUAL_DECISIONS = {"mismo", "distinto", "nuevo"}
EDITION_SEQUENCE = [*range(2014, 2022), 2023, 2024, 2025]
ROMAN_NUMBERS = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
}


@dataclass(frozen=True)
class CandidateMatch:
    score: float
    sort_score: float
    project_id: str
    best_row: pd.Series
    last_row: pd.Series
    pem_distance: float
    provisional: bool


def ensure_manual_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=MANUAL_COLUMNS).to_csv(path, index=False)
        return
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if "project_id_destino" in columns:
        pd.DataFrame(columns=MANUAL_COLUMNS).to_csv(path, index=False)
    elif columns != MANUAL_COLUMNS:
        raise ValueError(f"columnas inválidas en decisiones manuales: {path}")


def _manual_decisions(
    path: Path, rows: pd.DataFrame
) -> dict[tuple[int, str], dict[str, str]]:
    ensure_manual_file(path)
    manual = pd.read_csv(path, dtype=str).fillna("")
    decisions: dict[tuple[int, str], dict[str, str]] = {}
    for decision in manual.to_dict("records"):
        try:
            edition = int(decision["edicion"])
        except ValueError as error:
            raise ValueError("edición inválida en decisión manual") from error
        source_name = decision["proyecto_raw"]
        source_key = (edition, source_name)
        source_count = int(
            (rows["edicion"].eq(edition) & rows["proyecto_raw"].eq(source_name)).sum()
        )
        if source_count == 0:
            raise ValueError(
                f"origen manual inexistente: edición {edition}, proyecto {source_name!r}"
            )
        if source_count > 1 or source_key in decisions:
            raise ValueError(
                f"origen manual no único: edición {edition}, proyecto {source_name!r}"
            )
        action = decision["decision"]
        if action not in MANUAL_DECISIONS:
            raise ValueError(f"decisión manual inválida: {action!r}")
        if action in {"mismo", "distinto"}:
            try:
                target_edition = int(decision["destino_edicion"])
            except ValueError as error:
                raise ValueError("edición destino inválida en decisión manual") from error
            target_name = decision["destino_proyecto_raw"]
            if target_edition >= edition:
                raise ValueError(
                    "destino manual debe ser anterior: "
                    f"origen {edition}, destino {target_edition}"
                )
            target_count = int(
                (
                    rows["edicion"].eq(target_edition)
                    & rows["proyecto_raw"].eq(target_name)
                ).sum()
            )
            if target_count == 0:
                raise ValueError(
                    "destino manual inexistente: "
                    f"edición {target_edition}, proyecto {target_name!r}"
                )
            if target_count > 1:
                raise ValueError(
                    "destino manual no único: "
                    f"edición {target_edition}, proyecto {target_name!r}"
                )
        decisions[source_key] = decision
    return decisions


def _parenthetical_aliases(value: object) -> list[str]:
    text = str(value)
    stack: list[tuple[int, int]] = []
    contents: list[tuple[int, str]] = []
    for index, character in enumerate(text):
        if character == "(":
            stack.append((index, len(stack) + 1))
        elif character == ")" and stack:
            start, depth = stack.pop()
            contents.append((depth, text[start + 1 : index]))
    if not contents:
        return []
    selected = [min(contents, key=lambda item: item[0]), max(contents, key=lambda item: item[0])]
    aliases: list[str] = []
    for _, content in selected:
        content = re.sub(r"^\s*ex\b[\s:.-]*", "", content, flags=re.IGNORECASE)
        alias = normalize_name(content)
        if alias and alias not in aliases:
            aliases.append(alias)
    return aliases


def _names(row: pd.Series) -> list[str]:
    names: list[str] = []
    normalized = row.get("nombre_norm")
    if pd.notna(normalized) and str(normalized).strip():
        names.append(str(normalized))
    raw = row.get("proyecto_raw")
    if pd.notna(raw):
        main = normalize_name(raw)
        if main and main not in names:
            names.append(main)
        names.extend(alias for alias in _parenthetical_aliases(raw) if alias not in names)
    return names


def _name_similarity(source: pd.Series, candidate: pd.Series, scorer: Any) -> float:
    return max(
        (float(scorer(left, right)) for left in _names(source) for right in _names(candidate)),
        default=0.0,
    )


def _same_non_null(source: pd.Series, candidate: pd.Series, column: str) -> bool:
    left = source.get(column)
    right = candidate.get(column)
    return pd.notna(left) and pd.notna(right) and str(left) == str(right)


def _phase_marker(row: pd.Series) -> int | None:
    value = row.get("nombre_norm")
    if pd.isna(value):
        return None
    name = str(value).upper().strip()
    explicit = re.search(
        r"\b(?:FASE|ETAPA)\s+(\d+|VIII|VII|III|VI|IV|IX|II|V|I|X)\b",
        name,
    )
    marker = explicit.group(1) if explicit else None
    if marker is None:
        trailing = re.search(r"\b(VIII|VII|III|VI|IV|IX|II|V|I|X)$", name)
        marker = trailing.group(1) if trailing else None
    if marker is None:
        return None
    return int(marker) if marker.isdigit() else ROMAN_NUMBERS[marker]


def _score(source: pd.Series, candidate: pd.Series) -> float:
    score = _name_similarity(source, candidate, fuzz.token_set_ratio)
    if _same_non_null(source, candidate, "operador_norm"):
        score += 5
    if _same_non_null(source, candidate, "region"):
        score += 3
    source_phase = _phase_marker(source)
    candidate_phase = _phase_marker(candidate)
    if (
        source_phase is not None
        and candidate_phase is not None
        and source_phase != candidate_phase
    ):
        score = min(score, 70)
    return min(100.0, score)


def _token_sort_score(source: pd.Series, candidate: pd.Series) -> float:
    return _name_similarity(source, candidate, fuzz.token_sort_ratio)


def _new_id(number: int) -> str:
    return f"P{number:03d}"


def _commissioning_distance(source: pd.Series, candidate: pd.Series) -> float:
    source_value = source.get("puesta_en_marcha")
    candidate_value = candidate.get("puesta_en_marcha")
    if pd.isna(source_value) or pd.isna(candidate_value):
        return inf
    return abs(float(source_value) - float(candidate_value))


def _previous_data_edition(edition: int) -> int | None:
    previous = [candidate for candidate in EDITION_SEQUENCE if candidate < edition]
    return max(previous, default=None)


def _filter_provisional_candidates(
    candidates: list[CandidateMatch],
) -> list[CandidateMatch]:
    if not candidates:
        return []
    maximum = max(candidate.score for candidate in candidates)
    if any(
        not candidate.provisional
        and candidate.sort_score >= 85
        and candidate.score >= maximum - 2
        for candidate in candidates
    ):
        return [candidate for candidate in candidates if not candidate.provisional]
    return candidates


def _rank_key(candidate: CandidateMatch) -> tuple[float, int, float, float, str]:
    return (
        -candidate.score,
        -int(candidate.last_row["edicion"]),
        candidate.pem_distance,
        -candidate.sort_score,
        candidate.project_id,
    )


def _select_candidate(
    candidates: list[CandidateMatch],
) -> tuple[CandidateMatch | None, bool]:
    """Choose by score, recency, then commissioning; report a persistent tie."""
    if not candidates:
        return None, False
    maximum = max(candidate.score for candidate in candidates)
    tied = [candidate for candidate in candidates if maximum - candidate.score <= 0.5]
    latest = max(int(candidate.last_row["edicion"]) for candidate in tied)
    tied = [candidate for candidate in tied if int(candidate.last_row["edicion"]) == latest]
    closest = min(candidate.pem_distance for candidate in tied)
    tied = [candidate for candidate in tied if candidate.pem_distance == closest]
    if len(tied) == 1:
        return tied[0], False
    return None, True


def _is_continuation(source: pd.Series, candidate: CandidateMatch, edition: int) -> bool:
    if int(candidate.last_row["edicion"]) != _previous_data_edition(edition):
        return False
    identical = _same_non_null(source, candidate.last_row, "nombre_norm")
    source_phase = _phase_marker(source)
    candidate_phase = _phase_marker(candidate.last_row)
    phases_differ = (
        source_phase is not None
        and candidate_phase is not None
        and source_phase != candidate_phase
    )
    token_set = _name_similarity(source, candidate.last_row, fuzz.token_set_ratio)
    token_sort = _token_sort_score(source, candidate.last_row)
    return identical or (not phases_differ and token_set == 100 and token_sort >= 85)


def assign_project_ids(
    frame: pd.DataFrame,
    manual_path: Path,
    revision_path: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Assign stable IDs by comparing every row with all prior project histories."""
    rows = frame.copy().reset_index(drop=True)
    rows["_position"] = range(len(rows))
    rows = rows.sort_values(["edicion", "_position"], kind="stable").reset_index(drop=True)
    manual = _manual_decisions(manual_path, rows)
    histories: dict[str, list[pd.Series]] = {}
    anchor_ids: dict[tuple[int, str], str] = {}
    provisional_ids: set[str] = set()
    assigned: list[str] = []
    classes: list[str] = []
    revisions: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    next_number = 1

    for _, source in rows.iterrows():
        edition = int(source["edicion"])
        used_this_edition = {
            assigned[index]
            for index in range(len(assigned))
            if int(rows.iloc[index]["edicion"]) == edition
        }
        source_key = (edition, str(source["proyecto_raw"]))
        decision = manual.get(source_key)
        blocked: set[str] = set()
        force_new = False
        project_id: str | None = None
        classification = "nuevo"
        if decision is not None:
            applied.append(decision.copy())
            action = decision["decision"]
            if action in {"mismo", "distinto"}:
                target_key = (
                    int(decision["destino_edicion"]),
                    decision["destino_proyecto_raw"],
                )
                target = anchor_ids[target_key]
                if action == "mismo":
                    if target in used_this_edition:
                        raise ValueError(
                            f"destino manual ya usado en edición {edition}: {target_key!r}"
                        )
                    project_id = target
                    classification = "manual"
                else:
                    blocked.add(target)
            else:
                force_new = True
        ranked: list[CandidateMatch] = []
        for candidate_id, appearances in histories.items():
            if candidate_id in blocked or candidate_id in used_this_edition:
                continue
            comparable_appearances = appearances
            if _phase_marker(source) is not None:
                marked_appearances = [
                    candidate
                    for candidate in appearances
                    if _phase_marker(candidate) is not None
                ]
                if marked_appearances:
                    comparable_appearances = marked_appearances
            scored = [
                (_score(source, candidate), _token_sort_score(source, candidate), candidate)
                for candidate in comparable_appearances
            ]
            best_score, best_sort_score, best_row = max(
                scored,
                key=lambda item: (item[0], item[1], int(item[2]["edicion"])),
            )
            ranked.append(
                CandidateMatch(
                    score=best_score,
                    sort_score=best_sort_score,
                    project_id=candidate_id,
                    best_row=best_row,
                    last_row=appearances[-1],
                    pem_distance=_commissioning_distance(source, appearances[-1]),
                    provisional=candidate_id in provisional_ids,
                )
            )
        ranked = _filter_provisional_candidates(ranked)
        ranked.sort(key=_rank_key)
        if project_id is None and not force_new and ranked:
            top_score = ranked[0].score
            winner, tie = _select_candidate(ranked)
            if winner is not None and (
                _is_continuation(source, winner, edition)
                or (winner.score >= 92 and winner.sort_score >= 85)
            ):
                project_id = winner.project_id
                classification = "automatico"
            elif top_score >= 75 or tie:
                classification = "ambiguo"
                eligible = [candidate for candidate in ranked if candidate.score >= 60][:3]
                for rank, candidate_match in enumerate(eligible, start=1):
                    candidate_id = candidate_match.project_id
                    candidate = candidate_match.best_row
                    latest = candidate_match.last_row
                    revisions.append(
                        {
                            "rango": rank,
                            "edicion": edition,
                            "proyecto_raw": source["proyecto_raw"],
                            "candidato_project_id": candidate_id,
                            "candidato_nombre": candidate["proyecto_raw"],
                            "candidato_edicion": int(latest["edicion"]),
                            "candidato_proyecto_raw": latest["proyecto_raw"],
                            "ediciones_del_candidato": ",".join(
                                str(int(item["edicion"])) for item in histories[candidate_id]
                            ),
                            "puntaje": round(candidate_match.score, 1),
                            "operador": source.get("operador_norm"),
                            "region": source.get("region"),
                            "puesta_en_marcha": source.get("puesta_en_marcha"),
                            "inversion_mmusd": source.get("inversion_mmusd"),
                            "operador_candidato": candidate.get("operador_norm"),
                            "region_candidato": candidate.get("region"),
                            "puesta_en_marcha_candidato": candidate.get("puesta_en_marcha"),
                            "inversion_mmusd_candidato": candidate.get("inversion_mmusd"),
                            "decision": "",
                        }
                    )
        if project_id is None:
            project_id = _new_id(next_number)
            next_number += 1
            if classification == "ambiguo":
                provisional_ids.add(project_id)
        assigned.append(project_id)
        classes.append(classification)
        histories.setdefault(project_id, []).append(source)
        anchor_ids[source_key] = project_id

    rows["project_id"] = assigned
    rows["clasificacion_cruce"] = classes
    rows = rows.sort_values("_position", kind="stable").drop(columns="_position")
    revision = pd.DataFrame(revisions)
    revision_path.parent.mkdir(parents=True, exist_ok=True)
    if revision.empty:
        revision = pd.DataFrame(
            columns=[
                "rango",
                "edicion",
                "proyecto_raw",
                "candidato_project_id",
                "candidato_nombre",
                "candidato_edicion",
                "candidato_proyecto_raw",
                "ediciones_del_candidato",
                "puntaje",
                "operador",
                "region",
                "puesta_en_marcha",
                "inversion_mmusd",
                "operador_candidato",
                "region_candidato",
                "puesta_en_marcha_candidato",
                "inversion_mmusd_candidato",
                "decision",
            ]
        )
    revision.to_csv(revision_path, index=False)
    metrics = []
    for edition, group in rows.groupby("edicion", sort=True):
        counts = group["clasificacion_cruce"].value_counts()
        metrics.append(
            {
                "edicion": int(edition),
                "filas": len(group),
                "automaticos": int(counts.get("automatico", 0)),
                "automatico_pct": round(100 * counts.get("automatico", 0) / len(group), 1),
                "ambiguos": int(counts.get("ambiguo", 0)),
                "nuevos": int(counts.get("nuevo", 0)),
                "manuales": int(counts.get("manual", 0)),
            }
        )
    return rows.reset_index(drop=True), metrics, applied
