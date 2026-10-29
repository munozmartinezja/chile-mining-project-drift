# Preguntas difíciles

Las objeciones más fuertes que se le pueden hacer a este estudio, con lo que el repositorio permite responder hoy. Varias vienen de una revisión adversarial hecha antes de publicar.

### ¿Cómo sabe que dos nombres de ediciones distintas son el mismo proyecto?
Con un cruce difuso de nombre, alias y operador, más decisiones manuales documentadas en [`cruce_manual.csv`](../data/reference/cruce_manual.csv). Cada decisión manual declara su evidencia. {{n_vinculos_debiles|ent}} vínculos se apoyan solo en igual operador y capex o en la conciliación con Cochilco; son los más discutibles. La sensibilidad "sin vínculos débiles" los rompe todos: el coeficiente de Potencial queda en {{sens_min_ols_coef_potencial|d2}}–{{sens_max_ols_coef_potencial|d2}} entre escenarios y la brecha de operación entre grupos se mantiene.

### ¿No eligió las identidades para que cuadrara el cuadro de Cochilco?
Es un riesgo real cuando la conciliación justifica un vínculo. Por eso esos vínculos están marcados como débiles y se rompen en una sensibilidad. La conciliación no se fuerza: {{n_conciliaciones_con_ajuste|ent}} combinaciones cuadran solo con un ajuste, y cada ajuste nombra la partida que lo explica en [`ajustes_conciliacion.csv`](../data/reference/ajustes_conciliacion.csv).

### ¿Por qué cree en los desenlaces?
Porque no se infieren de la fecha: los declara Cochilco en su cuadro de cambios de cada edición. El {{pct_salidas_confirmadas_por_nombre|pct1}} de los egresos está confirmado por nombre en el texto del informe, con página y frase verificadas por el pipeline. Inferir el desenlace por la fecha anunciada habría errado en el {{pct_regla_fecha_discrepante|pct1}} de las salidas.

### ¿Y los {{n_egresos_por_cuadro|ent}} egresos sin mención por nombre?
Se identifican porque cuadran el conteo y el capex del cuadro oficial. Es la debilidad más importante del estudio. Si todos fueran salidas sin operar, la operación a 6 años de los proyectos Base o Probable bajaría de {{cif_op_6_base_probable|prop0}} a {{sens_min_cif_op_6_base_probable|prop0}}. Es un límite pesimista: ese escenario supone que todos están mal clasificados.

### La edición 2022 tiene su tabla en imagen. ¿Cómo la usó?
Se transcribió y se valida en cada corrida: los conteos y montos por condición y por sector coinciden exactamente con los subtotales que el informe publica en texto, y el total coincide con el declarado. Además hay un escenario de regresión que excluye todas las transiciones que tocan 2022.

### ¿Qué pasa si los proyectos censurados en realidad fracasaron, o en realidad operaron?
Son {{n_desenlace_censurado|ent}} proyectos (fusiones, cambios de metodología o salidas no declaradas). En los dos extremos, la operación a 6 años de Posible o Potencial queda entre {{sens_min_cif_op_6_posible_potencial|prop0}} y {{sens_max_cif_op_6_posible_potencial|prop0}}.

### ¿Cuántos proyectos quedan en seguimiento a 6 años?
Pocos: por eso cada incidencia acumulada se publica con su intervalo bootstrap al 95 %. Para Base o Probable, la operación a 6 años es {{cif_op_6_base_probable|prop0}} con intervalo {{ic_op_inf_6_base_probable|prop0}}–{{ic_op_sup_6_base_probable|prop0}}. El intervalo es ancho; la conclusión es la brecha entre grupos, no el nivel exacto.

### ¿Corrigió la truncación por la izquierda?
Las incidencias acumuladas se estiman sobre la cohorte incidente: proyectos que entran al catastro desde 2015 (n = {{n_cohorte_incidente|ent}}), observados desde su primera aparición. Los proyectos que ya estaban en 2014 solo entran en los indicadores descriptivos de deriva.

### ¿La condición predice el atraso o el desenlace?
El desenlace, sí: separa a los que llegan a operar. El atraso entre una revisión y la siguiente, no por sí sola: con solo la condición, el AUC es {{auc_agregado_solo_condicion|d2}}. El modelo completo, que agrega tamaño, horizonte anunciado y sector, llega a {{auc_agregado_completo|d2}}.

### Un año el modelo está en el azar. ¿Para qué sirve el AUC agregado?
Para resumir, no para esconder. El AUC por corte va de {{auc_corte_min|d2}} a {{auc_corte_max|d2}} y el tablero los muestra todos. La mejora en probabilidades frente a la tasa base no es concluyente: su intervalo va de {{mejora_brier_agregada_ic_inf|d3}} a {{mejora_brier_agregada_ic_sup|d3}}. El modelo ordena el riesgo; no da probabilidades confiables.

### ¿Hay fuga de información futura en la validación?
Las variables del modelo vienen de la edición previa a cada transición y el entrenamiento usa solo ediciones anteriores al corte. Lo que la validación no mide es el comportamiento en proyectos nuevos: el {{pct_test_proyectos_vistos|pct0}} de las transiciones de prueba es de proyectos que ya aparecían en el entrenamiento. El alcance es predecir la siguiente revisión de un proyecto que ya está en el catastro.

### ¿El precio del cobre causa los atrasos?
No se puede afirmar. Se agregó después de ver los datos, es igual para todos los proyectos de una edición y puede capturar tendencias del período. Se reporta como exploratorio.

### ¿Corrigió por comparaciones múltiples?
La familia principal es el modelo de deriva con condición, tamaño, horizonte y sector. Los demás modelos son exploratorios. El coeficiente de Potencial es positivo y significativo al 5 % en {{n_escenarios_potencial_positivo_p05|ent}} de {{n_escenarios_regresion|ent}} escenarios de regresión.

### ¿El capex del catastro mide sobrecostos?
No. Es el monto que se anuncia en cada edición, no el costo ejecutado. El {{pct_capex_sin_actualizar|pct1}} de las revisiones ni siquiera lo actualiza.

### ¿Las cifras están verificadas de verdad?
Cada cifra citada se recalcula con código que no importa el que la produce ({{n_cifras_verificadas|ent}} cifras). Eso atrapa errores de implementación. No atrapa un error de concepto compartido por ambos cálculos ni un error en las tablas curadas: para eso están la conciliación con Cochilco, las citas verificadas por página y las sensibilidades.

### ¿Puedo reproducirlo?
Sí. Las fuentes se descargan de sus URL públicas y se verifican por SHA-256; el pipeline completo corre con un comando y regenera este documento desde el registro. Las pruebas corren en integración continua sin datos locales.
