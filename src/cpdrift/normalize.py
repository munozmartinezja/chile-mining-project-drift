"""Pure catalog-based normalization of extracted project rows."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from .catalogos import (
    CONDICIONES,
    ETAPAS,
    OPERADOR_ALIASES,
    REGIONES,
    SECTORES,
    TIPOS_PROYECTO,
)


def _missing(value: object) -> bool:
    return value is None or bool(pd.isna(value))


def _ascii_key(value: object) -> str:
    if _missing(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"[A-Z0-9]+", text.upper()))


def normalize_name(value: object) -> str:
    """Normalize identity text and expand the approved abbreviations."""
    key = _ascii_key(value)
    replacements = {
        "AMP": "AMPLIACION",
        "MG": "MARGINAL",
        "EXP": "EXPANSION",
        "DIV": "DIVISION",
        "CIA": "COMPAÑIA",
    }
    return " ".join(replacements.get(token, token) for token in key.split())


def normalize_operator(value: object) -> str:
    key = normalize_name(value)
    if key.startswith(("CODELCO DIVISION ", "CODELCO DIV ")):
        return "CODELCO"
    suffixes = {"SA", "SCM", "LTDA", "SPA"}
    tokens = key.split()
    while tokens and tokens[-1] in suffixes:
        tokens.pop()
    stripped = " ".join(tokens)
    return OPERADOR_ALIASES.get(stripped, stripped)


def _catalog_key(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value).casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("–", "-").replace("—", "-").replace(".", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*([- /])\s*", r"\1", text)
    return text.replace("metalurgica s", "metalurgicas")


def _catalog_value(value: object, catalog: dict[str, str]) -> tuple[object, str | None]:
    if _missing(value) or not str(value).strip():
        return pd.NA, None
    literal = re.sub(r"\s+", " ", str(value)).strip()
    key = _catalog_key(literal)
    normalized_catalog = {_catalog_key(source): target for source, target in catalog.items()}
    if key in normalized_catalog:
        return normalized_catalog[key], None
    return literal, literal


def normalize_region(value: object) -> object:
    if _missing(value) or not str(value).strip():
        return pd.NA
    repaired = re.sub(r"(?<=[a-záéíóúñ])\s+(?=[a-záéíóúñ]\b)", "", str(value), flags=re.IGNORECASE)
    key = _ascii_key(repaired)
    return REGIONES.get(key, pd.NA)


def normalize_projects(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Normalize source values, rejecting only unknown condition/region labels."""
    result = frame.copy()
    result["condicion"] = result["condicion"].map(
        lambda value: _ascii_key(value) if not _missing(value) else ""
    )
    unknown_conditions = sorted(set(result["condicion"]) - set(CONDICIONES) - {""})
    if unknown_conditions:
        raise ValueError(
            "condicion desconocida: " + ", ".join(unknown_conditions)
        )
    result["condicion"] = result["condicion"].map(
        lambda value: CONDICIONES.get(value, pd.NA)
    )
    result["region"] = result["region"].map(normalize_region)
    result["nombre_norm"] = result["proyecto_raw"].map(normalize_name)
    result["operador_norm"] = result["operador_raw"].map(normalize_operator)

    unknowns: dict[str, list[str]] = {}
    for column, catalog in (
        ("sector", SECTORES),
        ("tipo_proyecto", TIPOS_PROYECTO),
        ("etapa", ETAPAS),
    ):
        values = result[column].map(lambda value, c=catalog: _catalog_value(value, c))
        result[column] = values.map(lambda pair: pair[0])
        unknowns[column] = sorted(
            {pair[1] for pair in values if pair[1] is not None}
        )
    return result, unknowns


def impute_regions(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill a missing region only from another appearance of the same project."""
    result = frame.copy()
    result["region_imputada"] = False
    known = (
        result.dropna(subset=["region"])
        .groupby("project_id", sort=False)["region"]
        .agg(lambda values: values.iloc[0] if values.nunique() == 1 else pd.NA)
    )
    missing = result["region"].isna()
    fill = result.loc[missing, "project_id"].map(known)
    result.loc[missing, "region"] = fill
    result.loc[missing & result["region"].notna(), "region_imputada"] = True
    return result
