"""Human-readable register of the headline indicator figures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .indicators import IndicatorTables


def _stat(series: pd.Series, operation: str) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(getattr(values, operation)())


def _percentage(series: pd.Series, comparison: str) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(getattr(values, comparison)(0).mean() * 100)


def build_claims_register(tables: IndicatorTables) -> list[dict[str, object]]:
    """Build the auditable list of required headline figures."""
    transitions = tables.transiciones
    reliability = tables.fiabilidad
    entries = tables.entradas_salidas
    values = {
        "n_proyectos": len(tables.proyectos_deriva),
        "n_filas_agregadas_excluidas": tables.n_filas_agregadas_excluidas,
        "n_transiciones": len(transitions),
        "media_delta_pem": _stat(transitions["delta_pem"], "mean"),
        "mediana_delta_pem": _stat(transitions["delta_pem"], "median"),
        "pct_transiciones_atraso": _percentage(transitions["delta_pem"], "gt"),
        "pct_transiciones_adelanto": _percentage(transitions["delta_pem"], "lt"),
        "media_tasa_deriva": _stat(transitions["tasa_deriva"], "mean"),
        "mediana_delta_capex_real_pct": _stat(
            transitions["delta_capex_real_pct"], "median"
        ),
        "pct_fiabilidad_a_tiempo": reliability["pct_a_tiempo"],
        "pct_fiabilidad_1_anio": reliability["pct_hasta_1_anio"],
        "mediana_deriva_acumulada_operados": reliability[
            "mediana_deriva_acumulada"
        ],
        "n_salidas_sin_operar": int(entries["salidas_sin_operar"].sum()),
        "n_salidas_operacion": int(entries["salidas_operacion"].sum()),
    }
    metadata = {
        "n_proyectos": ("Proyectos no agregados", "proyectos", "Conteo de project_id no agregados."),
        "n_filas_agregadas_excluidas": (
            "Filas agregadas excluidas",
            "filas",
            "Conteo de apariciones cuyo nombre identifica un agregado.",
        ),
        "n_transiciones": (
            "Transiciones entre apariciones",
            "transiciones",
            "Conteo de pares consecutivos dentro de cada project_id.",
        ),
        "media_delta_pem": (
            "Atraso medio entre apariciones",
            "años",
            "Media de pem_next menos pem_prev.",
        ),
        "mediana_delta_pem": (
            "Atraso mediano entre apariciones",
            "años",
            "Mediana de pem_next menos pem_prev.",
        ),
        "pct_transiciones_atraso": (
            "Transiciones con atraso",
            "porcentaje",
            "Porcentaje de transiciones con delta_pem mayor que cero.",
        ),
        "pct_transiciones_adelanto": (
            "Transiciones con adelanto",
            "porcentaje",
            "Porcentaje de transiciones con delta_pem menor que cero.",
        ),
        "media_tasa_deriva": (
            "Tasa media de deriva",
            "años por año",
            "Media de delta_pem dividida por gap_anios.",
        ),
        "mediana_delta_capex_real_pct": (
            "Variación mediana del capex real",
            "porcentaje",
            "Mediana de la variación porcentual del capex real 2025.",
        ),
        "pct_fiabilidad_a_tiempo": (
            "Fiabilidad del año a tiempo",
            "porcentaje",
            "Porcentaje de operados con deriva acumulada menor o igual a cero.",
        ),
        "pct_fiabilidad_1_anio": (
            "Fiabilidad dentro de un año",
            "porcentaje",
            "Porcentaje de operados con deriva acumulada menor o igual a un año.",
        ),
        "mediana_deriva_acumulada_operados": (
            "Deriva mediana de proyectos operados",
            "años",
            "Mediana de deriva acumulada entre operados con dos o más apariciones.",
        ),
        "n_salidas_sin_operar": (
            "Salidas sin operar",
            "salidas",
            "Suma de salidas cuyo PEM previo es posterior a la edición siguiente.",
        ),
        "n_salidas_operacion": (
            "Salidas a operación",
            "salidas",
            "Suma de salidas cuyo PEM previo no supera la edición siguiente.",
        ),
    }
    return [
        {
            "id": claim_id,
            "descripcion": metadata[claim_id][0],
            "valor": value,
            "unidad": metadata[claim_id][1],
            "calculo": metadata[claim_id][2],
        }
        for claim_id, value in values.items()
    ]
