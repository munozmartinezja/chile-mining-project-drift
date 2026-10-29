# Método

Este documento describe cómo se construyó el estudio, siguiendo las etapas de CRISP-DM. Todas las cifras provienen del [registro verificado](registro_cifras.csv); cada una se recalcula con código independiente del que la produce.

## 1. Pregunta

Entre una edición y la siguiente de la cartera de Cochilco, ¿cuánto se corre la fecha de puesta en marcha anunciada y cuánto cambia el capex de cada proyecto? ¿Qué proyectos llegan a operar y cuáles salen de la cartera sin operar? Son las mismas preguntas de control de proyectos sobre desviación de plazo y de costo, aplicadas a la cartera pública de inversión minera.

## 2. Fuentes

- **Cartera de proyectos de Cochilco**, {{n_ediciones|ent}} ediciones anuales consecutivas (2014 a 2025). Cada edición trae una tabla por proyecto con fecha de puesta en marcha, capex, condición (Base, Probable, Posible o Potencial), región, sector y etapa. Cada edición tiene entre {{min_proyectos_edicion|ent}} y {{max_proyectos_edicion|ent}} proyectos.
- **Edición 2022:** su tabla de proyectos es una imagen. Se transcribió ({{n_filas_transcritas_2022|ent}} filas) y la transcripción se valida en el pipeline: los conteos y montos por condición y por sector coinciden exactamente con los subtotales que el mismo informe publica en texto, y el total coincide con el declarado.
- **Precio del cobre:** Banco Mundial, Pink Sheet. Promedio de enero a junio del año de cada edición, en US$ de 2025.
- **Deflactor:** CPI-U de EE. UU., promedio semestral. El capex real se expresa en US$ de 2025.

Cada archivo fuente tiene su URL y su SHA-256 en [`fuentes_raw.csv`](../data/reference/fuentes_raw.csv). El pipeline se detiene si un archivo no coincide con su hash.

## 3. Preparación de datos

**Extracción.** Las tablas se leen de los PDF y, en 2025, del anexo en Excel. El total extraído de cada edición se compara con el total que declara el informe, citado por página.

**Filas agregadas.** Las partidas que agrupan varias inversiones (por ejemplo, "Otros proyectos de desarrollo") no son proyectos: se excluyen ({{n_filas_agregadas_excluidas|ent}} filas).

**Identidad entre ediciones.** El mismo proyecto cambia de nombre, de operador o de fase entre ediciones. El cruce combina una comparación difusa de nombres, alias y operador con decisiones manuales documentadas. Cada decisión manual declara su evidencia: nombre común, texto de Cochilco, mismo operador y capex, o solo conciliación. Las dos últimas se consideran vínculos débiles; hay {{n_vinculos_debiles|ent}}, y una sensibilidad los rompe para medir su efecto. El resultado son {{n_proyectos|ent}} proyectos distintos y {{n_transiciones|ent}} transiciones entre apariciones consecutivas.

**Desenlaces declarados, no inferidos.** Cada salida de la cartera se clasifica según lo que Cochilco declara en su cuadro de cambios: entró en operación, salió por reestructuración, salió por cambio de metodología o se fusionó con otro proyecto. Hay {{n_egresos_declarados|ent}} egresos declarados. El {{pct_salidas_confirmadas_por_nombre|pct1}} está confirmado por nombre en el texto del informe; los {{n_egresos_por_cuadro|ent}} restantes se identifican porque cuadran el conteo y el capex del cuadro oficial, sin mención individual. La alternativa, inferir el desenlace por la fecha anunciada, habría clasificado mal el {{pct_regla_fecha_discrepante|pct1}} de las salidas.

**Conciliación con Cochilco.** Para cada par de ediciones consecutivas, las entradas y salidas observadas se comparan con el cuadro oficial de cambios (conteo y capex). En los {{n_pares_conciliados|ent}} pares de ediciones, {{n_conciliaciones_exactas|ent}} combinaciones de par y categoría cuadran de forma exacta y {{n_conciliaciones_con_ajuste|ent}} con un ajuste documentado y nombrado (por ejemplo, cuando el cuadro valora un proyecto con su capex actualizado, o cuenta dos fases como un solo proyecto). El pipeline no avanza si una combinación no cuadra.

**Umbral.** Cochilco cambió el umbral de inclusión entre ediciones. Los conteos de salidas se reportan también sobre un umbral homogéneo de 90 MMUS$, y una sensibilidad restringe todo el análisis a proyectos con capex inicial de al menos 90 MMUS$.

## 4. Modelado

**Deriva.** Para cada transición se mide el cambio del año de puesta en marcha anunciado y del capex. La deriva acumulada es el año anunciado en cada edición menos el anunciado al entrar.

**Desenlaces: riesgos competitivos.** Entrar en operación y salir sin operar son eventos que compiten. Se estima la incidencia acumulada con Aalen–Johansen sobre la cohorte incidente: proyectos que entran al catastro desde 2015 (n = {{n_cohorte_incidente|ent}}), para que todos se observen desde su primera aparición. Los proyectos vigentes en 2025 y los fusionados o retirados por metodología se tratan como censurados. Los intervalos al 95 % se obtienen por bootstrap de proyectos con semilla fija.

**Regresión asociativa.** Mínimos cuadrados ordinarios de la tasa de deriva (años de atraso por año transcurrido) sobre condición, log del capex, horizonte anunciado y sector, con errores agrupados por proyecto ({{ols_n|ent}} transiciones). Un modelo ampliado agrega el precio del cobre y la propiedad estatal. El precio del cobre se incorporó después de ver los datos: se reporta como exploratorio.

**Predicción del atraso.** Logit penalizado de Firth para la probabilidad de que la siguiente revisión atrase la fecha. Se valida con origen móvil: se entrena con las ediciones anteriores a cada corte y se prueba en el corte ({{n_cortes|ent}} cortes, {{n_test_agregado|ent}} transiciones de prueba). Se comparan el AUC, el puntaje de Brier frente a la tasa base y la calibración.

## 5. Evaluación

**Registro verificado.** Cada cifra citada se recalcula en un módulo que no importa el código que la produce. Hoy hay {{n_cifras_verificadas|ent}} cifras verificadas. La verificación detecta errores de implementación; no detecta errores conceptuales compartidos ni errores en las tablas curadas. Para eso están la conciliación con Cochilco y las sensibilidades.

**Sensibilidades.** Siete escenarios: el principal, los censurados tratados como salida o como operación, los egresos identificados por cuadre tratados como operación o como salida, sin los vínculos de identidad débiles y solo proyectos sobre 90 MMUS$.

| Métrica a 6 años | Principal | Rango entre escenarios |
|---|---|---|
| Operación, Base o Probable | {{cif_op_6_base_probable|prop0}} | {{sens_min_cif_op_6_base_probable|prop0}}–{{sens_max_cif_op_6_base_probable|prop0}} |
| Salida sin operar, Base o Probable | {{cif_sal_6_base_probable|prop0}} | {{sens_min_cif_sal_6_base_probable|prop0}}–{{sens_max_cif_sal_6_base_probable|prop0}} |
| Operación, Posible o Potencial | {{cif_op_6_posible_potencial|prop0}} | {{sens_min_cif_op_6_posible_potencial|prop0}}–{{sens_max_cif_op_6_posible_potencial|prop0}} |
| Salida sin operar, Posible o Potencial | {{cif_sal_6_posible_potencial|prop0}} | {{sens_min_cif_sal_6_posible_potencial|prop0}}–{{sens_max_cif_sal_6_posible_potencial|prop0}} |
| Coeficiente OLS de Potencial (años por año) | {{ols_coef_potencial|d2}} | {{sens_min_ols_coef_potencial|d2}}–{{sens_max_ols_coef_potencial|d2}} |

La brecha entre grupos se mantiene en todos los escenarios. El nivel de operación de Base o Probable es sensible a los egresos identificados por cuadre: si todos fueran salidas sin operar, bajaría a {{sens_min_cif_op_6_base_probable|prop0}}. Es un límite pesimista, no una estimación alternativa.

## 6. Resultados

- **Deriva:** el {{pct_transiciones_atraso|pct1}} de las revisiones atrasa la fecha, el {{pct_transiciones_sin_cambio|pct1}} no la cambia y el {{pct_transiciones_adelanto|pct1}} la adelanta. Al año 4, la deriva acumulada mediana es {{mediana_deriva_anio4|ent}} años (percentil 90: {{p90_deriva_anio4|ent}}; {{n_proyectos_anio4|ent}} proyectos en seguimiento). El {{pct_fiabilidad_a_tiempo|pct1}} de los proyectos que operaron lo hizo en el año anunciado al entrar y el {{pct_fiabilidad_1_anio|pct1}} dentro de un año.
- **Capex:** el {{pct_capex_sin_actualizar|pct1}} de las revisiones no actualiza el monto. Entre las que sí lo hacen, la variación mediana es {{mediana_delta_capex_nom_pct_actualizados|pct1}} nominal y {{mediana_delta_capex_real_pct_actualizados|pct1}} real. El capex del catastro es un monto anunciado, no un costo ejecutado.
- **Atraso por condición:** atrasa la fecha el {{pct_atraso_base|pct1}} de las revisiones de proyectos Base, el {{pct_atraso_probable|pct1}} de los Probable, el {{pct_atraso_posible|pct1}} de los Posible y el {{pct_atraso_potencial|pct1}} de los Potencial.
- **Regresión:** frente a Base, la condición Potencial se asocia con {{ols_coef_potencial|d2}} años más de atraso por año transcurrido. El tamaño (log del capex: {{ols_coef_log_capex|d2}}) y un horizonte anunciado más corto (coeficiente {{ols_coef_horizonte|d2}}) también se asocian con más atraso. En el modelo ampliado, el precio del cobre tiene coeficiente {{ols_amp_coef_precio_cu|d2}} (p {{ols_amp_p_precio_cu|p}}), un resultado exploratorio. La propiedad estatal no muestra diferencia (p = {{ols_amp_p_estatal|d2}}).
- **Validación:** AUC agregado {{auc_agregado_completo|d2}} (IC 95 %: {{auc_agregado_completo_ic_inf|d2}}–{{auc_agregado_completo_ic_sup|d2}}); por corte entre {{auc_corte_min|d2}} y {{auc_corte_max|d2}}. La condición sola no ordena el riesgo (AUC {{auc_agregado_solo_condicion|d2}}). La mejora del Brier frente a la tasa base es {{mejora_brier_agregada|d3}} con intervalo de {{mejora_brier_agregada_ic_inf|d3}} a {{mejora_brier_agregada_ic_sup|d3}}: no es concluyente. En promedio, el modelo sobreestima la tasa de atraso en {{calibracion_agregada_completo|pp0}}.

## 7. Limitaciones

1. **Asociación, no causalidad.** Ningún coeficiente identifica un efecto causal. La condición la asigna Cochilco con información que también anticipa el desenlace: el estudio valida esa clasificación, no la descubre.
2. **Capex anunciado.** El monto del catastro no es el costo real; el estudio nunca habla de sobrecosto.
3. **"Entró en operación" lo declara Cochilco.** No hay verificación física independiente.
4. **Colas pequeñas.** A 6 años quedan pocos proyectos en seguimiento por grupo; por eso cada incidencia acumulada se reporta con su intervalo.
5. **Egresos por cuadre e identidad manual.** {{n_egresos_por_cuadro|ent}} egresos y {{n_vinculos_debiles|ent}} vínculos de identidad no tienen respaldo individual por nombre; las sensibilidades acotan su efecto.
6. **Edición 2022 transcrita.** Se valida contra subtotales del informe, pero no proviene de texto extraído automáticamente.
7. **Validación del modelo.** Mide cómo predice la siguiente revisión de proyectos que ya están en la cartera, no el comportamiento de proyectos nuevos. Su desempeño cambia de un año a otro.
8. **Precio del cobre.** Agregado después de ver los datos; es una variable común a todos los proyectos de una edición y puede capturar tendencias temporales.
