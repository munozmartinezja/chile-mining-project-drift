"""Explicit publication labels for coded dashboard fields."""

from __future__ import annotations

import pandas as pd

SECTOR_LABELS = {
    "GRAN_MINERIA_CU": "Gran minería del cobre",
    "MEDIANA_MINERIA_CU": "Mediana minería del cobre",
    "ESTATAL_CU": "Minería estatal del cobre",
    "PLANTAS_METALURGICAS": "Plantas metalúrgicas",
    "MINERIA_INDUSTRIAL": "Minería industrial",
    "Oro": "Oro",
    "Hierro": "Hierro",
    "Litio": "Litio",
}
CONDITION_LABELS = {"BASE": "Base", "PROBABLE": "Probable", "POSIBLE": "Posible", "POTENCIAL": "Potencial"}
CONDITION_ORDER = {"BASE": 1, "PROBABLE": 2, "POSIBLE": 3, "POTENCIAL": 4}
OUTCOME_LABELS = {"entro_en_operacion": "Entró en operación", "salio_sin_operar": "Salió sin operar", "censurado": "Sin desenlace observable", "vigente": "En cartera 2025"}
OUTCOME_ORDER = {"entro_en_operacion": 1, "salio_sin_operar": 2, "censurado": 3, "vigente": 4}
SOURCE_LABELS = {"nombre": "Confirmado por nombre", "cuadro": "Confirmado por cuadre de capex", "revision": "Revisión documentada", "vigente": "En cartera 2025"}
GROUP_LABELS = {"BASE_PROBABLE": "Base o probable", "POSIBLE_POTENCIAL": "Posible o potencial", "TOTAL": "Total"}
CATEGORY_LABELS = {"entraron_operacion": "Entraron en operación", "reestructuracion": "Salieron por reestructuración", "nuevos": "Ingresaron", "metodologia": "Salieron por metodología"}


def _map_required(series: pd.Series, mapping: dict[str, object], field: str) -> pd.Series:
    mapped = series.map(mapping)
    unknown = sorted(series.loc[series.notna() & mapped.isna()].astype(str).unique())
    if unknown:
        raise ValueError(f"valor sin etiqueta en {field}: {unknown[0]}")
    return mapped


def add_export_labels(table_name: str, table: pd.DataFrame) -> pd.DataFrame:
    """Add labels required by one exported table, rejecting unknown codes."""
    result = table.copy()
    condition_column = next((item for item in ("condicion_inicial", "condicion_prev", "condicion") if item in result), None)
    if "sector" in result:
        result["sector_etiqueta"] = _map_required(result["sector"], SECTOR_LABELS, "sector")
    if condition_column:
        result["condicion_etiqueta"] = _map_required(result[condition_column], CONDITION_LABELS, condition_column)
        result["condicion_orden"] = _map_required(result[condition_column], CONDITION_ORDER, condition_column)
    if "desenlace" in result:
        result["desenlace_etiqueta"] = _map_required(result["desenlace"], OUTCOME_LABELS, "desenlace")
        result["desenlace_orden"] = _map_required(result["desenlace"], OUTCOME_ORDER, "desenlace")
    if "fuente_desenlace" in result:
        result["fuente_etiqueta"] = _map_required(result["fuente_desenlace"], SOURCE_LABELS, "fuente_desenlace")
    if "grupo" in result:
        result["grupo_etiqueta"] = _map_required(result["grupo"], GROUP_LABELS, "grupo")
    if "categoria" in result and table_name != "desc_perfil":
        result["categoria_etiqueta"] = _map_required(result["categoria"], CATEGORY_LABELS, "categoria")
    if table_name == "desc_perfil":
        mask = result["dimension"].eq("condicion_inicial")
        result["condicion_etiqueta"] = pd.NA
        result["condicion_orden"] = pd.NA
        result.loc[mask, "condicion_etiqueta"] = _map_required(result.loc[mask, "categoria"], CONDITION_LABELS, "condicion_inicial")
        result.loc[mask, "condicion_orden"] = _map_required(result.loc[mask, "categoria"], CONDITION_ORDER, "condicion_inicial")
    return result
