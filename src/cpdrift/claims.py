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
    validacion_cortes: pd.DataFrame | None = None,
    validacion_agregada: pd.DataFrame | None = None,
    validacion_bootstrap: dict[str, float] | None = None,
    descriptivas: dict[str, pd.DataFrame] | None = None,
    visuales: dict[str, pd.DataFrame] | None = None,
    sensibilidad: pd.DataFrame | None = None,
    sensibilidad_publicable: pd.DataFrame | None = None,
    regla_fecha: pd.DataFrame | None = None,
    transcripcion_2022: pd.DataFrame | None = None,
    cruce_manual: pd.DataFrame | None = None,
    validacion_datos: pd.DataFrame | None = None,
    validacion_predicciones: pd.DataFrame | None = None,
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
        exact_reconciliations = (
            conciliacion["delta_n"].eq(0)
            & conciliacion["compara_capex"].eq(True)
            & conciliacion["delta_capex"].eq(0)
        )
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
                "n_conciliaciones_exactas": int(exact_reconciliations.sum()),
                "n_conciliaciones_con_ajuste": int(
                    (~exact_reconciliations).sum()
                ),
            }
        )
    if transcripcion_2022 is not None:
        values["n_filas_transcritas_2022"] = len(transcripcion_2022)
    if cruce_manual is not None:
        values["n_vinculos_debiles"] = int(
            (
                cruce_manual["decision"].eq("mismo")
                & cruce_manual["evidencia"].isin({"operador_capex", "conciliacion"})
            ).sum()
        )
    if egresos is not None:
        values.update(
            {
                "n_egresos_declarados": len(egresos),
                "n_egresos_por_cuadro": int(egresos["fuente"].eq("cuadro").sum()),
            }
        )
    if validacion_datos is not None and validacion_predicciones is not None:
        seen = sum(
            row.project_id
            in set(
                validacion_datos.loc[
                    validacion_datos["ed_prev"].lt(row.corte), "project_id"
                ]
            )
            for row in validacion_predicciones.itertuples(index=False)
        )
        values["pct_test_proyectos_vistos"] = (
            None
            if validacion_predicciones.empty
            else float(seen / len(validacion_predicciones) * 100)
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
                "auc_corte_unico_completo": float(complete["auc"]),
                "auc_corte_unico_solo_condicion": float(condition["auc"]),
                "brier_corte_unico_completo": float(complete["brier"]),
                "brier_corte_unico_prevalencia": float(prevalence["brier"]),
                "calibracion_corte_unico_completo": float(
                    complete["calibracion_global"]
                ),
                "n_train_corte_unico": int(complete["n_train"]),
                "n_test_corte_unico": int(complete["n_test"]),
            }
        )
    if (
        validacion_cortes is not None
        and validacion_agregada is not None
        and validacion_bootstrap is not None
    ):
        aggregate = validacion_agregada.set_index("modelo")
        complete = aggregate.loc["completo"]
        prevalence = aggregate.loc["prevalencia"]
        values.update(
            {
                "auc_agregado_completo": float(complete["auc"]),
                "auc_agregado_completo_ic_inf": validacion_bootstrap["auc_ic_inf"],
                "auc_agregado_completo_ic_sup": validacion_bootstrap["auc_ic_sup"],
                "auc_agregado_solo_condicion": float(
                    aggregate.loc["solo_condicion", "auc"]
                ),
                "brier_agregado_completo": float(complete["brier"]),
                "brier_agregado_prevalencia": float(prevalence["brier"]),
                "mejora_brier_agregada": float(
                    prevalence["brier"] - complete["brier"]
                ),
                "mejora_brier_agregada_ic_inf": validacion_bootstrap[
                    "mejora_brier_ic_inf"
                ],
                "mejora_brier_agregada_ic_sup": validacion_bootstrap[
                    "mejora_brier_ic_sup"
                ],
                "calibracion_agregada_completo": float(complete["calibracion"]),
                "auc_corte_min": float(
                    validacion_cortes.loc[
                        validacion_cortes["modelo"].eq("completo"), "auc"
                    ].min()
                ),
                "auc_corte_max": float(
                    validacion_cortes.loc[
                        validacion_cortes["modelo"].eq("completo"), "auc"
                    ].max()
                ),
                "n_cortes": int(validacion_cortes["corte"].nunique()),
                "n_test_agregado": int(complete["n_test"]),
            }
        )
    if descriptivas is not None:
        portfolio = descriptivas["desc_cartera_edicion"]
        profile = descriptivas["desc_perfil"]
        sector = profile.loc[profile["dimension"].eq("sector")]
        region = profile.loc[profile["dimension"].eq("region")]
        total_projects = int(
            descriptivas["desc_desenlace_cohorte"]["n"].sum()
        )
        copper = sector.loc[
            sector["categoria"].isin(
                {"GRAN_MINERIA_CU", "ESTATAL_CU", "MEDIANA_MINERIA_CU"}
            ),
            "n_proyectos",
        ].sum()
        ant = region.loc[
            region["categoria"].astype(str).str.upper().eq("ANTOFAGASTA"),
            "n_proyectos",
        ].sum()
        values.update(
            {
                **{
                    f"pct_atraso_{condition.lower()}": (
                        None
                        if transitions.loc[
                            transitions["condicion_prev"].eq(condition), "delta_pem"
                        ].dropna().empty
                        else float(
                            transitions.loc[
                                transitions["condicion_prev"].eq(condition),
                                "delta_pem",
                            ]
                            .dropna()
                            .gt(0)
                            .mean()
                            * 100
                        )
                    )
                    for condition in ("BASE", "PROBABLE", "POSIBLE", "POTENCIAL")
                },
                "pct_transiciones_sin_cambio": (
                    None
                    if transitions["delta_pem"].dropna().empty
                    else float(transitions["delta_pem"].dropna().eq(0).mean() * 100)
                ),
                "mediana_capex_inicial_real": _stat(
                    tables.proyectos_deriva["capex_inicial_real"], "median"
                ),
                "pct_proyectos_cobre": (
                    None if total_projects == 0 else float(copper / total_projects * 100)
                ),
                "pct_proyectos_antofagasta": (
                    None if total_projects == 0 else float(ant / total_projects * 100)
                ),
                "n_ediciones": int(portfolio["edicion"].nunique()),
                "min_proyectos_edicion": int(portfolio["n_proyectos"].min()),
                "max_proyectos_edicion": int(portfolio["n_proyectos"].max()),
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
                "n_escenarios_potencial_positivo_p05": int(
                    sum(
                        coefficients.loc[item] > 0 and p_values.loc[item] < 0.05
                        for item in common
                    )
                ),
            }
        )
    if sensibilidad_publicable is not None:
        scenario_names = ", ".join(sensibilidad_publicable["escenario"].astype(str))
        for metric in (
            "cif_op_6_base_probable",
            "cif_sal_6_base_probable",
            "cif_op_6_posible_potencial",
            "cif_sal_6_posible_potencial",
            "ols_coef_potencial",
        ):
            values[f"sens_min_{metric}"] = float(
                sensibilidad_publicable[metric].min()
            )
            values[f"sens_max_{metric}"] = float(
                sensibilidad_publicable[metric].max()
            )
    if regla_fecha is not None:
        values["pct_regla_fecha_discrepante"] = (
            None
            if regla_fecha.empty
            else float(regla_fecha["discrepante"].mean() * 100)
        )
    if visuales is not None:
        year_four = visuales["viz_abanico"].loc[
            visuales["viz_abanico"]["anios_desde_primera"].eq(4)
        ]
        values.update(
            {
                "mediana_deriva_anio4": None if year_four.empty else float(year_four.iloc[0]["mediana"]),
                "p90_deriva_anio4": None if year_four.empty else float(year_four.iloc[0]["p90"]),
                "n_proyectos_anio4": None if year_four.empty else int(year_four.iloc[0]["n_proyectos"]),
            }
        )
    metadata = {
        "n_proyectos": ("Proyectos no agregados", "proyectos", "Conteo de project_id no agregados."),
        "n_filas_agregadas_excluidas": (
            "Filas agregadas excluidas",
            "proyectos",
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
            "proyectos",
            "Suma de salidas por reestructuración con capex nominal previo de al menos 90.",
        ),
        "n_salidas_umbral90_operacion": (
            "Salidas a operación sobre umbral homogéneo",
            "proyectos",
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
            "pares de ediciones",
            "Conteo de pares distintos en la conciliación de desenlaces.",
        ),
        "n_conciliaciones_exactas": (
            "Combinaciones conciliadas sin ajuste",
            "combinaciones",
            "Filas par × categoría donde lo observado coincide con el cuadro oficial sin ajuste documentado (conteo y capex).",
        ),
        "n_conciliaciones_con_ajuste": (
            "Combinaciones conciliadas con ajuste documentado",
            "combinaciones",
            "Filas par × categoría con ajuste en ajustes_conciliacion.csv (conteo o capex no comparable).",
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
        "auc_corte_unico_completo": ("AUC temporal modelo completo, corte único", "AUC", "AUC Mann-Whitney en prueba."),
        "auc_corte_unico_solo_condicion": ("AUC temporal solo condición, corte único", "AUC", "AUC Mann-Whitney en prueba."),
        "brier_corte_unico_completo": ("Brier temporal modelo completo, corte único", "puntaje de Brier", "Media del error cuadrático probabilístico."),
        "brier_corte_unico_prevalencia": ("Brier temporal prevalencia, corte único", "puntaje de Brier", "Media del error cuadrático probabilístico."),
        "calibracion_corte_unico_completo": ("Calibración modelo completo, corte único", "proporción", "Media predicha menos tasa observada."),
        "n_train_corte_unico": ("Entrenamiento del corte único", "transiciones", "Conteo con ed_prev hasta 2019."),
        "n_test_corte_unico": ("Prueba del corte único", "transiciones", "Conteo con ed_prev desde 2020."),
        "n_escenarios_regresion": ("Escenarios de regresión", "escenarios", "Conteo de escenarios OLS ampliados."),
        "n_escenarios_potencial_positivo_p05": (
            "Escenarios de regresión con coeficiente Potencial positivo y p < 0,05",
            "escenarios",
            "Conteo entre principal, sin 2022 y sin empresas estatales.",
        ),
        "pct_regla_fecha_discrepante": ("Discrepancia de regla de fecha", "porcentaje", "Salidas cuya regla de fecha contradice lo declarado."),
        "pct_atraso_base": ("Transiciones Base con atraso", "porcentaje", "Proporción de transiciones Base con aumento del año previsto."),
        "pct_atraso_probable": ("Transiciones Probable con atraso", "porcentaje", "Proporción de transiciones Probable con aumento del año previsto."),
        "pct_atraso_posible": ("Transiciones Posible con atraso", "porcentaje", "Proporción de transiciones Posible con aumento del año previsto."),
        "pct_atraso_potencial": ("Transiciones Potencial con atraso", "porcentaje", "Proporción de transiciones Potencial con aumento del año previsto."),
        "pct_transiciones_sin_cambio": ("Transiciones sin cambio de fecha", "porcentaje", "Proporción con delta de puesta en marcha igual a cero."),
        "mediana_capex_inicial_real": ("Capex inicial real mediano", "MMUS$ de 2025", "Mediana del capex inicial deflactado."),
        "pct_proyectos_cobre": ("Proyectos del sector cobre", "porcentaje", "Proporción de proyectos en sectores de cobre."),
        "pct_proyectos_antofagasta": ("Proyectos ubicados en Antofagasta", "porcentaje", "Proporción de proyectos con región Antofagasta."),
        "n_ediciones": ("Ediciones de cartera analizadas", "ediciones", "Conteo de ediciones presentes en el panel."),
        "min_proyectos_edicion": ("Mínimo de proyectos por edición", "proyectos", "Mínimo del conteo anual no agregado."),
        "max_proyectos_edicion": ("Máximo de proyectos por edición", "proyectos", "Máximo del conteo anual no agregado."),
        "auc_agregado_completo": ("AUC agregado del modelo completo", "AUC", "AUC sobre predicciones fuera de muestra concatenadas."),
        "auc_agregado_completo_ic_inf": ("Límite inferior del AUC agregado", "AUC", "Percentil 2,5 del bootstrap por proyecto."),
        "auc_agregado_completo_ic_sup": ("Límite superior del AUC agregado", "AUC", "Percentil 97,5 del bootstrap por proyecto."),
        "auc_agregado_solo_condicion": ("AUC agregado usando solo condición", "AUC", "AUC fuera de muestra del modelo reducido."),
        "brier_agregado_completo": ("Brier agregado del modelo completo", "puntaje de Brier", "Error cuadrático probabilístico agregado."),
        "brier_agregado_prevalencia": ("Brier agregado del modelo de prevalencia", "puntaje de Brier", "Error cuadrático probabilístico del referente."),
        "mejora_brier_agregada": ("Mejora agregada del puntaje de Brier", "puntaje de Brier", "Brier de prevalencia menos Brier completo."),
        "mejora_brier_agregada_ic_inf": ("Límite inferior de la mejora de Brier", "puntaje de Brier", "Percentil 2,5 del bootstrap por proyecto."),
        "mejora_brier_agregada_ic_sup": ("Límite superior de la mejora de Brier", "puntaje de Brier", "Percentil 97,5 del bootstrap por proyecto."),
        "calibracion_agregada_completo": ("Calibración agregada del modelo completo", "proporción", "Probabilidad media menos frecuencia observada."),
        "auc_corte_min": ("AUC mínimo entre cortes temporales", "AUC", "Mínimo del AUC del modelo completo entre cortes."),
        "auc_corte_max": ("AUC máximo entre cortes temporales", "AUC", "Máximo del AUC del modelo completo entre cortes."),
        "n_cortes": ("Cortes de validación temporal", "cortes", "Conteo de cortes sucesivos evaluados."),
        "n_test_agregado": ("Observaciones agregadas de prueba", "transiciones", "Conteo de predicciones fuera de muestra agregadas."),
        "n_filas_transcritas_2022": (
            "Filas transcritas de la cartera 2022",
            "filas",
            "Conteo de filas en cartera_2022_transcrita.csv.",
        ),
        "n_vinculos_debiles": (
            "Vínculos manuales con evidencia débil",
            "vínculos",
            "Decisiones mismo sustentadas por operador y capex o conciliación.",
        ),
        "n_egresos_declarados": (
            "Egresos declarados",
            "egresos",
            "Conteo de filas en egresos_declarados.csv.",
        ),
        "n_egresos_por_cuadro": (
            "Egresos identificados por cuadro",
            "egresos",
            "Conteo de egresos cuya fuente es cuadro.",
        ),
        "pct_test_proyectos_vistos": (
            "Transiciones de prueba de proyectos vistos en entrenamiento",
            "porcentaje",
            "Porcentaje agregado sobre los siete cortes de origen móvil.",
        ),
        "mediana_deriva_anio4": ("Deriva mediana al cuarto año", "años", "Mediana entre proyectos observados al cuarto año."),
        "p90_deriva_anio4": ("Percentil 90 de deriva al cuarto año", "años", "Percentil 90 entre proyectos observados al cuarto año."),
        "n_proyectos_anio4": ("Proyectos observados al cuarto año", "proyectos", "Conteo que sustenta el abanico al cuarto año."),
    }
    for metric, description, unit in (
        ("cif_op_6_base_probable", "CIF de operación a seis años en Base/Probable", "proporción"),
        ("cif_sal_6_base_probable", "CIF de salida a seis años en Base/Probable", "proporción"),
        ("cif_op_6_posible_potencial", "CIF de operación a seis años en Posible/Potencial", "proporción"),
        ("cif_sal_6_posible_potencial", "CIF de salida a seis años en Posible/Potencial", "proporción"),
        ("ols_coef_potencial", "coeficiente OLS de condición Potencial", "años por año"),
    ):
        if sensibilidad_publicable is not None:
            calculation = f"Extremo entre los siete escenarios: {scenario_names}."
            metadata[f"sens_min_{metric}"] = (
                f"Mínimo de sensibilidad: {description}", unit, calculation
            )
            metadata[f"sens_max_{metric}"] = (
                f"Máximo de sensibilidad: {description}", unit, calculation
            )
    missing_metadata = sorted(set(values) - set(metadata))
    if missing_metadata:
        raise ValueError("cifras sin metadatos: " + ", ".join(missing_metadata))
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
        for row in cif.itertuples(index=False):
            for column in interval_columns:
                value = getattr(row, column)
                public_id = f"{column}_{row.t}_{str(row.grupo).lower()}"
                verified_interval = (
                    row.cohorte == "incidente"
                    and row.grupo in {"BASE_PROBABLE", "POSIBLE_POTENCIAL"}
                    and row.t in {4, 6}
                )
                claims.append(
                    {
                        "id": public_id if verified_interval else f"{column}_{row.cohorte}_{row.grupo}_{row.t}",
                        "descripcion": (
                            f"Intervalo bootstrap {column} para {row.cohorte}, "
                            f"{row.grupo}, t={row.t}"
                        ),
                        "valor": None if pd.isna(value) else float(value),
                        "unidad": "proporción",
                        "calculo": "Percentil bootstrap por proyecto.",
                        **({} if verified_interval else {"estado": "informativa"}),
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
    verified_count = 1 + sum(
        item.get("estado") != "informativa" for item in claims
    )
    claims.append(
        {
            "id": "n_cifras_verificadas",
            "descripcion": "Cifras verificadas",
            "valor": verified_count,
            "unidad": "cifras",
            "calculo": "Conteo de cifras verificables, incluida esta misma.",
        }
    )
    return claims
