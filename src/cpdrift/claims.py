"""Human-readable register of the headline indicator figures."""

from __future__ import annotations

import re
import unicodedata
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


def _cif_value(
    table: pd.DataFrame, group: str, column: str, time: int = 6
) -> float | None:
    row = table.loc[
        table["cohorte"].eq("incidente")
        & table["grupo"].eq(group)
        & table["t"].eq(time)
    ]
    return None if row.empty else float(row.iloc[0][column])


def _ols_value(table: pd.DataFrame, term: str) -> float | None:
    row = table.loc[table["termino"].eq(term)]
    return None if row.empty else float(row.iloc[0]["coef"])


def _table_value(
    table: pd.DataFrame, row_column: str, row_value: str, value_column: str
) -> float | None:
    row = table.loc[table[row_column].eq(row_value)]
    return None if row.empty else float(row.iloc[0][value_column])


def _snake_ascii(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]+", "_", text)).strip("_").lower()


def build_claims_register(
    tables: IndicatorTables,
    cif: pd.DataFrame | None = None,
    ols: pd.DataFrame | None = None,
    logit: pd.DataFrame | None = None,
    projects: pd.DataFrame | None = None,
    egresos: pd.DataFrame | None = None,
    conciliacion: pd.DataFrame | None = None,
    ols_ampliado: pd.DataFrame | None = None,
    validacion: pd.DataFrame | None = None,
    sensibilidad: pd.DataFrame | None = None,
    regla_fecha: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    """Build the auditable list of required headline figures."""
    transitions = tables.transiciones
    reliability = tables.fiabilidad
    entries = tables.entradas_salidas
    comparable_capex = transitions.loc[
        transitions["capex_prev_nom"].notna()
        & transitions["capex_next_nom"].notna()
    ]
    updated_capex = comparable_capex.loc[
        comparable_capex["capex_prev_nom"].ne(comparable_capex["capex_next_nom"])
    ]
    values: dict[str, object] = {
        "n_proyectos": len(tables.proyectos_deriva),
        "n_filas_agregadas_excluidas": tables.n_filas_agregadas_excluidas,
        "n_transiciones": len(transitions),
        "media_delta_pem": _stat(transitions["delta_pem"], "mean"),
        "mediana_delta_pem": _stat(transitions["delta_pem"], "median"),
        "pct_transiciones_atraso": _percentage(transitions["delta_pem"], "gt"),
        "pct_transiciones_adelanto": _percentage(transitions["delta_pem"], "lt"),
        "media_tasa_deriva": _stat(transitions["tasa_deriva"], "mean"),
        "pct_capex_sin_actualizar": (
            None
            if comparable_capex.empty
            else float(
                comparable_capex["capex_prev_nom"]
                .eq(comparable_capex["capex_next_nom"])
                .mean()
                * 100
            )
        ),
        "mediana_delta_capex_nom_pct_actualizados": _stat(
            updated_capex["delta_capex_nom_pct"], "median"
        ),
        "mediana_delta_capex_real_pct_actualizados": _stat(
            updated_capex["delta_capex_real_pct"], "median"
        ),
        "pct_fiabilidad_a_tiempo": reliability["pct_a_tiempo"],
        "pct_fiabilidad_1_anio": reliability["pct_hasta_1_anio"],
        "mediana_deriva_acumulada_operados": reliability[
            "mediana_deriva_acumulada"
        ],
        "n_salidas_umbral90_sin_operar": int(entries["salidas_sin_operar"].sum()),
        "n_salidas_umbral90_operacion": int(entries["salidas_operacion"].sum()),
    }
    if projects is not None and egresos is not None and conciliacion is not None:
        outcomes = projects["desenlace"].value_counts()
        values.update(
            {
                "n_desenlace_operacion": int(outcomes.get("entro_en_operacion", 0)),
                "n_desenlace_sin_operar": int(outcomes.get("salio_sin_operar", 0)),
                "n_desenlace_censurado": int(outcomes.get("censurado", 0)),
                "pct_salidas_confirmadas_por_nombre": (
                    None
                    if egresos.empty
                    else float(egresos["fuente"].eq("nombre").mean() * 100)
                ),
                "n_pares_conciliados": int(
                    conciliacion[["edicion_desde", "edicion_hasta"]]
                    .drop_duplicates()
                    .shape[0]
                ),
            }
        )
    if cif is not None and ols is not None:
        incident_total = cif.loc[
            cif["cohorte"].eq("incidente") & cif["grupo"].eq("TOTAL")
        ]
        values.update(
            {
                "n_cohorte_incidente": (
                    0
                    if incident_total.empty
                    else int(incident_total["en_riesgo"].max())
                ),
                "cif_op_6_base_probable": _cif_value(
                    cif, "BASE_PROBABLE", "cif_op"
                ),
                "cif_sal_6_base_probable": _cif_value(
                    cif, "BASE_PROBABLE", "cif_sal"
                ),
                "cif_op_6_posible_potencial": _cif_value(
                    cif, "POSIBLE_POTENCIAL", "cif_op"
                ),
                "cif_sal_6_posible_potencial": _cif_value(
                    cif, "POSIBLE_POTENCIAL", "cif_sal"
                ),
                "ols_coef_potencial": _ols_value(
                    ols,
                    'C(condicion_prev, Treatment("BASE"))[T.POTENCIAL]',
                ),
                "ols_coef_log_capex": _ols_value(ols, "log_capex"),
                "ols_coef_horizonte": _ols_value(ols, "horizonte"),
                "ols_n": 0 if ols.empty else int(ols.iloc[0]["n"]),
            }
        )
    if cif is not None:
        values.update(
            {
                f"cif_{event}_4_{group_id}": _cif_value(
                    cif, group, f"cif_{event}", 4
                )
                for group_id, group in (
                    ("base_probable", "BASE_PROBABLE"),
                    ("posible_potencial", "POSIBLE_POTENCIAL"),
                )
                for event in ("op", "sal")
            }
        )
    if ols_ampliado is not None:
        potential = 'C(condicion_prev, Treatment("BASE"))[T.POTENCIAL]'
        values.update(
            {
                "ols_amp_coef_potencial": _table_value(
                    ols_ampliado, "termino", potential, "coef"
                ),
                "ols_amp_coef_precio_cu": _table_value(
                    ols_ampliado, "termino", "log_precio_cu", "coef"
                ),
                "ols_amp_p_precio_cu": _table_value(
                    ols_ampliado, "termino", "log_precio_cu", "p"
                ),
                "ols_amp_coef_estatal": _table_value(
                    ols_ampliado, "termino", "estatal", "coef"
                ),
                "ols_amp_p_estatal": _table_value(
                    ols_ampliado, "termino", "estatal", "p"
                ),
            }
        )
    if validacion is not None:
        complete = validacion.loc[validacion["modelo"].eq("completo")].iloc[0]
        condition = validacion.loc[
            validacion["modelo"].eq("solo_condicion")
        ].iloc[0]
        prevalence = validacion.loc[
            validacion["modelo"].eq("prevalencia")
        ].iloc[0]
        values.update(
            {
                "auc_test_completo": float(complete["auc"]),
                "auc_test_solo_condicion": float(condition["auc"]),
                "brier_test_completo": float(complete["brier"]),
                "brier_test_prevalencia": float(prevalence["brier"]),
                "calibracion_test_completo": float(
                    complete["calibracion_global"]
                ),
                "n_train": int(complete["n_train"]),
                "n_test": int(complete["n_test"]),
            }
        )
    if sensibilidad is not None:
        regression = sensibilidad.loc[sensibilidad["modelo"].eq("ols_ampliado")]
        potential = 'C(condicion_prev, Treatment("BASE"))[T.POTENCIAL]'
        coefficients = regression.loc[
            regression["metrica"].eq(f"coef:{potential}")
        ].set_index("escenario")["valor"]
        p_values = regression.loc[
            regression["metrica"].eq(f"p:{potential}")
        ].set_index("escenario")["valor"]
        common = coefficients.index.intersection(p_values.index)
        values.update(
            {
                "n_escenarios_regresion": int(regression["escenario"].nunique()),
                "n_escenarios_potencial_robusto": int(
                    sum(
                        coefficients.loc[item] > 0 and p_values.loc[item] < 0.05
                        for item in common
                    )
                ),
            }
        )
    if regla_fecha is not None:
        values["pct_regla_fecha_discrepante"] = (
            None
            if regla_fecha.empty
            else float(regla_fecha["discrepante"].mean() * 100)
        )
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
        "pct_capex_sin_actualizar": (
            "Transiciones sin actualización nominal de capex",
            "porcentaje",
            "Porcentaje con capex nominal final igual al inicial.",
        ),
        "mediana_delta_capex_nom_pct_actualizados": (
            "Variación mediana del capex nominal actualizado",
            "porcentaje",
            "Mediana nominal solo donde el capex nominal cambia.",
        ),
        "mediana_delta_capex_real_pct_actualizados": (
            "Variación mediana del capex real actualizado",
            "porcentaje",
            "Mediana real solo donde el capex nominal cambia.",
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
        "n_salidas_umbral90_sin_operar": (
            "Salidas sin operar sobre umbral homogéneo",
            "salidas",
            "Suma de salidas por reestructuración con capex nominal previo de al menos 90.",
        ),
        "n_salidas_umbral90_operacion": (
            "Salidas a operación sobre umbral homogéneo",
            "salidas",
            "Suma de salidas a operación con capex nominal previo de al menos 90.",
        ),
        "n_desenlace_operacion": (
            "Proyectos que entraron en operación",
            "proyectos",
            "Conteo de desenlaces declarados como operación.",
        ),
        "n_desenlace_sin_operar": (
            "Proyectos que salieron sin operar",
            "proyectos",
            "Conteo de desenlaces declarados como reestructuración.",
        ),
        "n_desenlace_censurado": (
            "Proyectos con salida censurada",
            "proyectos",
            "Conteo de desenlaces declarados como censurados.",
        ),
        "pct_salidas_confirmadas_por_nombre": (
            "Salidas confirmadas por nombre",
            "porcentaje",
            "Filas con fuente nombre sobre el total de egresos declarados.",
        ),
        "n_pares_conciliados": (
            "Pares de ediciones conciliados",
            "pares",
            "Conteo de pares distintos en la conciliación de desenlaces.",
        ),
        "n_cohorte_incidente": (
            "Proyectos de la cohorte incidente",
            "proyectos",
            "Conteo con primera edición desde 2015 y tiempo positivo.",
        ),
        "cif_op_6_base_probable": (
            "CIF de operación a seis años en BASE_PROBABLE",
            "proporción",
            "Estimación discreta para la cohorte incidente.",
        ),
        "cif_sal_6_base_probable": (
            "CIF de salida a seis años en BASE_PROBABLE",
            "proporción",
            "Estimación discreta para la cohorte incidente.",
        ),
        "cif_op_6_posible_potencial": (
            "CIF de operación a seis años en POSIBLE_POTENCIAL",
            "proporción",
            "Estimación discreta para la cohorte incidente.",
        ),
        "cif_sal_6_posible_potencial": (
            "CIF de salida a seis años en POSIBLE_POTENCIAL",
            "proporción",
            "Estimación discreta para la cohorte incidente.",
        ),
        "ols_coef_potencial": (
            "Coeficiente OLS para condición POTENCIAL",
            "años por año",
            "Coeficiente respecto de BASE en el modelo asociativo.",
        ),
        "ols_coef_log_capex": (
            "Coeficiente OLS para log_capex",
            "años por año",
            "Coeficiente de log_capex en el modelo asociativo.",
        ),
        "ols_coef_horizonte": (
            "Coeficiente OLS para horizonte",
            "años por año",
            "Coeficiente de horizonte en el modelo asociativo.",
        ),
        "ols_n": (
            "Observaciones del modelo OLS",
            "transiciones",
            "Conteo de casos completos usados por OLS.",
        ),
        "cif_op_4_base_probable": ("CIF operación a cuatro años, BASE_PROBABLE", "proporción", "Estimación discreta incidente."),
        "cif_sal_4_base_probable": ("CIF salida a cuatro años, BASE_PROBABLE", "proporción", "Estimación discreta incidente."),
        "cif_op_4_posible_potencial": ("CIF operación a cuatro años, POSIBLE_POTENCIAL", "proporción", "Estimación discreta incidente."),
        "cif_sal_4_posible_potencial": ("CIF salida a cuatro años, POSIBLE_POTENCIAL", "proporción", "Estimación discreta incidente."),
        "ols_amp_coef_potencial": ("Coeficiente potencial OLS ampliado", "años por año", "Coeficiente asociativo ampliado."),
        "ols_amp_coef_precio_cu": ("Coeficiente precio cobre OLS ampliado", "años por año", "Coeficiente de log_precio_cu."),
        "ols_amp_p_precio_cu": ("p precio cobre OLS ampliado", "probabilidad", "Valor p agrupado."),
        "ols_amp_coef_estatal": ("Coeficiente estatal OLS ampliado", "años por año", "Coeficiente del indicador estatal."),
        "ols_amp_p_estatal": ("p estatal OLS ampliado", "probabilidad", "Valor p agrupado."),
        "auc_test_completo": ("AUC temporal modelo completo", "proporción", "AUC Mann-Whitney en prueba."),
        "auc_test_solo_condicion": ("AUC temporal solo condición", "proporción", "AUC Mann-Whitney en prueba."),
        "brier_test_completo": ("Brier temporal modelo completo", "puntaje", "Media del error cuadrático probabilístico."),
        "brier_test_prevalencia": ("Brier temporal prevalencia", "puntaje", "Media del error cuadrático probabilístico."),
        "calibracion_test_completo": ("Calibración global modelo completo", "proporción", "Media predicha menos tasa observada."),
        "n_train": ("Observaciones de entrenamiento", "transiciones", "Conteo con ed_prev hasta 2019."),
        "n_test": ("Observaciones de prueba", "transiciones", "Conteo con ed_prev desde 2020."),
        "n_escenarios_regresion": ("Escenarios de regresión", "escenarios", "Conteo de escenarios OLS ampliados."),
        "n_escenarios_potencial_robusto": ("Escenarios con asociación potencial robusta", "escenarios", "Coeficiente positivo con p menor a 0,05."),
        "pct_regla_fecha_discrepante": ("Discrepancia de regla de fecha", "porcentaje", "Salidas cuya regla de fecha contradice lo declarado."),
    }
    claims = [
        {
            "id": claim_id,
            "descripcion": metadata[claim_id][0],
            "valor": value,
            "unidad": metadata[claim_id][1],
            "calculo": metadata[claim_id][2],
        }
        for claim_id, value in values.items()
    ]
    if cif is not None:
        interval_columns = ("ic_op_inf", "ic_op_sup", "ic_sal_inf", "ic_sal_sup")
        for row in cif.loc[cif["t"].isin((2, 4, 6))].itertuples(index=False):
            for column in interval_columns:
                value = getattr(row, column)
                claims.append(
                    {
                        "id": f"{column}_{row.cohorte}_{row.grupo}_{row.t}",
                        "descripcion": (
                            f"Intervalo bootstrap {column} para {row.cohorte}, "
                            f"{row.grupo}, t={row.t}"
                        ),
                        "valor": None if pd.isna(value) else float(value),
                        "unidad": "proporción",
                        "calculo": "Percentil bootstrap por proyecto.",
                        "estado": "informativa",
                    }
                )
    if logit is not None:
        for row in logit.itertuples(index=False):
            if str(row.termino).lower() == "intercept":
                continue
            claims.append(
                {
                    "id": f"logit_or_{_snake_ascii(row.termino)}",
                    "descripcion": f"Razón de odds para {row.termino}",
                    "valor": float(row.coef),
                    "unidad": "razón de odds",
                    "calculo": (
                        f"Modelo asociativo; IC 95 % [{row.ic_inf}, {row.ic_sup}]."
                    ),
                    "estado": "informativa",
                }
            )
    return claims
