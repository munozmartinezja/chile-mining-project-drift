# Brief · Deriva de la cartera de inversión minera en Chile (2014–2025)

## En una línea
Los proyectos que Cochilco clasifica como Base o Probable llegan a operar mucho más que los Posible o Potencial, y la fecha anunciada se posterga en la mitad de las revisiones anuales.

## En lenguaje simple
Cada año, Cochilco publica la lista de grandes proyectos mineros que se esperan en Chile, con la fecha en que deberían empezar a operar, cuánto costarían y qué tan probable es que se concreten. Este estudio siguió a {{n_proyectos|ent}} de esos proyectos durante {{n_ediciones|ent}} ediciones de la lista para ver qué pasó con ellos.

- **La etiqueta de Cochilco sirve.** De cada 10 proyectos que entraron a la lista como "Base" o "Probable", unos {{cif_op_6_base_probable|de10}} estaban operando 6 años después. De los "Posible" o "Potencial", solo unos {{cif_op_6_posible_potencial|de10}}.
- **Las fechas se corren.** En la mitad de las actualizaciones anuales ({{pct_transiciones_atraso|pct0}}), la fecha de inicio se posterga. Cuatro años después de aparecer, el proyecto típico que sigue en la lista lleva {{mediana_deriva_anio4|ent}} años de atraso.
- **Se puede anticipar quién se atrasa, pero no con precisión.** Un modelo entrenado con años anteriores ordena qué proyectos tienen más riesgo de atrasarse mejor que el azar, aunque no todos los años y sin dar probabilidades confiables.

Ninguno de estos resultados dice por qué ocurre. Son patrones, no causas.

## Capa técnica
- **Datos:** {{n_ediciones|ent}} ediciones de la cartera de Cochilco, {{n_proyectos|ent}} proyectos, {{n_transiciones|ent}} transiciones entre ediciones. Edición 2022 transcrita y validada contra los subtotales del informe.
- **Desenlaces:** declarados por Cochilco y conciliados con su cuadro de cambios en {{n_pares_conciliados|ent}} pares de ediciones ({{n_conciliaciones_exactas|ent}} combinaciones exactas y {{n_conciliaciones_con_ajuste|ent}} con ajuste documentado).
- **Riesgos competitivos (Aalen–Johansen, cohorte incidente n = {{n_cohorte_incidente|ent}}):** operación a 6 años {{cif_op_6_base_probable|prop0}} (IC 95 % {{ic_op_inf_6_base_probable|prop0}}–{{ic_op_sup_6_base_probable|prop0}}) en Base o Probable frente a {{cif_op_6_posible_potencial|prop0}} ({{ic_op_inf_6_posible_potencial|prop0}}–{{ic_op_sup_6_posible_potencial|prop0}}) en Posible o Potencial.
- **Deriva (OLS, errores agrupados por proyecto):** la condición Potencial se asocia con {{ols_coef_potencial|d2}} años más de atraso por año transcurrido que Base; rango entre sensibilidades {{sens_min_ols_coef_potencial|d2}}–{{sens_max_ols_coef_potencial|d2}}.
- **Predicción (logit de Firth, validación de origen móvil, {{n_cortes|ent}} cortes):** AUC {{auc_agregado_completo|d2}} (IC 95 % {{auc_agregado_completo_ic_inf|d2}}–{{auc_agregado_completo_ic_sup|d2}}); por corte {{auc_corte_min|d2}}–{{auc_corte_max|d2}}. Mejora del Brier no concluyente.
- **Rigor:** {{n_cifras_verificadas|ent}} cifras con recálculo independiente; 7 escenarios de sensibilidad; fuentes verificadas por SHA-256.
- **Principal limitación:** {{n_egresos_por_cuadro|ent}} egresos se identifican por cuadre de capex y no por nombre; en el escenario más pesimista, la operación a 6 años de Base o Probable baja a {{sens_min_cif_op_6_base_probable|prop0}}.

Detalle: [método](metodo.md) · [preguntas difíciles](preguntas_dificiles.md) · [repositorio](../README.md).
