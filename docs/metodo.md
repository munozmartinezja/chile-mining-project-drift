<!-- Generado desde docs/plantillas con cpdrift.publicacion. No editar a mano. -->
# Método

Este documento describe cómo se construyó el estudio, siguiendo las etapas de CRISP-DM. Todas las cifras provienen del [registro verificado](registro_cifras.csv); cada una se recalcula con código independiente del que la produce.

## 1. Pregunta

Entre una edición y la siguiente de la cartera de Cochilco, ¿cuánto se corre la fecha de puesta en marcha anunciada y cuánto cambia el capex de cada proyecto? ¿Qué proyectos llegan a operar y cuáles salen de la cartera sin operar? Son las mismas preguntas de control de proyectos sobre desviación de plazo y de costo, aplicadas a la cartera pública de inversión minera.

## 2. Fuentes

- **Cartera de proyectos de Cochilco**, 12 ediciones anuales consecutivas (2014 a 2025). Cada edición trae una tabla por proyecto con fecha de puesta en marcha, capex, condición (Base, Probable, Posible o Potencial), región, sector y etapa. Cada edición tiene entre 35 y 57 proyectos.
- **Edición 2022:** su tabla de proyectos es una imagen. Se transcribió (53 filas) y la transcripción se valida en el pipeline: los conteos y montos por condición y por sector coinciden exactamente con los subtotales que el mismo informe publica en texto, y el total coincide con el declarado.
- **Precio del cobre:** Banco Mundial, Pink Sheet. Promedio de enero a junio del año de cada edición, en US$ de 2025.
- **Deflactor:** CPI-U de EE. UU., promedio semestral. El capex real se expresa en US$ de 2025.

Cada archivo fuente tiene su URL y su SHA-256 en [`fuentes_raw.csv`](../data/reference/fuentes_raw.csv). El pipeline se detiene si un archivo no coincide con su hash.

## 3. Preparación de datos

**Extracción.** Las tablas se leen de los PDF y, en 2025, del anexo en Excel. El total extraído de cada edición se compara con el total que declara el informe, citado por página.

**Filas agregadas.** Las partidas que agrupan varias inversiones (por ejemplo, "Otros proyectos de desarrollo") no son proyectos: se excluyen (27 filas).

**Identidad entre ediciones.** El mismo proyecto cambia de nombre, de operador o de fase entre ediciones. El cruce combina una comparación difusa de nombres, alias y operador con decisiones manuales documentadas. Cada decisión manual declara su evidencia: nombre común, texto de Cochilco, mismo operador y capex, o solo conciliación. Las dos últimas se consideran vínculos débiles; hay 22, y una sensibilidad los rompe para medir su efecto. El resultado son 133 proyectos distintos y 409 transiciones entre apariciones consecutivas.

**Desenlaces declarados, no inferidos.** Cada salida de la cartera se clasifica según lo que Cochilco declara en su cuadro de cambios: entró en operación, salió por reestructuración, salió por cambio de metodología o se fusionó con otro proyecto. Hay 81 egresos declarados. El 75,3 % está confirmado por nombre en el texto del informe; los 18 restantes se identifican porque cuadran el conteo y el capex del cuadro oficial, sin mención individual. La alternativa, inferir el desenlace por la fecha anunciada, habría clasificado mal el 7,8 % de las salidas.

**Conciliación con Cochilco.** Para cada par de ediciones consecutivas, las entradas y salidas observadas se comparan con el cuadro oficial de cambios (conteo y capex). En los 11 pares de ediciones, 28 combinaciones de par y categoría cuadran de forma exacta y 5 con un ajuste documentado y nombrado (por ejemplo, cuando el cuadro valora un proyecto con su capex actualizado, o cuenta dos fases como un solo proyecto). El pipeline no avanza si una combinación no cuadra.

**Umbral.** Cochilco cambió el umbral de inclusión entre ediciones. Los conteos de salidas se reportan también sobre un umbral homogéneo de 90 MMUS$, y una sensibilidad restringe todo el análisis a proyectos con capex inicial de al menos 90 MMUS$.

## 4. Modelado

**Deriva.** Para cada transición se mide el cambio del año de puesta en marcha anunciado y del capex. La deriva acumulada es el año anunciado en cada edición menos el anunciado al entrar.

**Desenlaces: riesgos competitivos.** Entrar en operación y salir sin operar son eventos que compiten. Se estima la incidencia acumulada con Aalen–Johansen sobre la cohorte incidente: proyectos que entran al catastro desde 2015 (n = 64), para que todos se observen desde su primera aparición. Los proyectos vigentes en 2025 y los fusionados o retirados por metodología se tratan como censurados. Los intervalos al 95 % se obtienen por bootstrap de proyectos con semilla fija.

**Regresión asociativa.** Mínimos cuadrados ordinarios de la tasa de deriva (años de atraso por año transcurrido) sobre condición, log del capex, horizonte anunciado y sector, con errores agrupados por proyecto (409 transiciones). Un modelo ampliado agrega el precio del cobre y la propiedad estatal. El precio del cobre se incorporó después de ver los datos: se reporta como exploratorio.

**Predicción del atraso.** Logit penalizado de Firth para la probabilidad de que la siguiente revisión atrase la fecha. Se valida con origen móvil: se entrena con las ediciones anteriores a cada corte y se prueba en el corte (7 cortes, 275 transiciones de prueba). Se comparan el AUC, el puntaje de Brier frente a la tasa base y la calibración.

## 5. Evaluación

**Registro verificado.** Cada cifra citada se recalcula en un módulo que no importa el código que la produce. Hoy hay 111 cifras verificadas. La verificación detecta errores de implementación; no detecta errores conceptuales compartidos ni errores en las tablas curadas. Para eso están la conciliación con Cochilco y las sensibilidades.

**Sensibilidades.** Siete escenarios: el principal, los censurados tratados como salida o como operación, los egresos identificados por cuadre tratados como operación o como salida, sin los vínculos de identidad débiles y solo proyectos sobre 90 MMUS$.

| Métrica a 6 años | Principal | Rango entre escenarios |
|---|---|---|
| Operación, Base o Probable | 73 % | 52 %–74 % |
| Salida sin operar, Base o Probable | 10 % | 10 %–31 % |
| Operación, Posible o Potencial | 23 % | 21 %–31 % |
| Salida sin operar, Posible o Potencial | 34 % | 31 %–41 % |
| Coeficiente OLS de Potencial (años por año) | 1,16 | 1,05–1,19 |

La brecha entre grupos se mantiene en todos los escenarios. El nivel de operación de Base o Probable es sensible a los egresos identificados por cuadre: si todos fueran salidas sin operar, bajaría a 52 %. Es un límite pesimista, no una estimación alternativa.

## 6. Resultados

- **Deriva:** el 50,1 % de las revisiones atrasa la fecha, el 46,0 % no la cambia y el 3,9 % la adelanta. Al año 4, la deriva acumulada mediana es 3 años (percentil 90: 5; 45 proyectos en seguimiento). El 45,9 % de los proyectos que operaron lo hizo en el año anunciado al entrar y el 67,6 % dentro de un año.
- **Capex:** el 69,2 % de las revisiones no actualiza el monto. Entre las que sí lo hacen, la variación mediana es 8,8 % nominal y 5,5 % real. El capex del catastro es un monto anunciado, no un costo ejecutado.
- **Atraso por condición:** atrasa la fecha el 38,9 % de las revisiones de proyectos Base, el 49,2 % de los Probable, el 54,2 % de los Posible y el 59,7 % de los Potencial.
- **Regresión:** frente a Base, la condición Potencial se asocia con 1,16 años más de atraso por año transcurrido. El tamaño (log del capex: 0,27) y un horizonte anunciado más corto (coeficiente -0,27) también se asocian con más atraso. En el modelo ampliado, el precio del cobre tiene coeficiente 1,13 (p < 0,001), un resultado exploratorio. La propiedad estatal no muestra diferencia (p = 0,25).
- **Validación:** AUC agregado 0,66 (IC 95 %: 0,59–0,73); por corte entre 0,50 y 0,82. La condición sola no ordena el riesgo (AUC 0,50). La mejora del Brier frente a la tasa base es 0,020 con intervalo de -0,003 a 0,043: no es concluyente. En promedio, el modelo sobreestima la tasa de atraso en 6 pp.

## 7. Limitaciones

1. **Asociación, no causalidad.** Ningún coeficiente identifica un efecto causal. La condición la asigna Cochilco con información que también anticipa el desenlace: el estudio valida esa clasificación, no la descubre.
2. **Capex anunciado.** El monto del catastro no es el costo real; el estudio nunca habla de sobrecosto.
3. **"Entró en operación" lo declara Cochilco.** No hay verificación física independiente.
4. **Colas pequeñas.** A 6 años quedan pocos proyectos en seguimiento por grupo; por eso cada incidencia acumulada se reporta con su intervalo.
5. **Egresos por cuadre e identidad manual.** 18 egresos y 22 vínculos de identidad no tienen respaldo individual por nombre; las sensibilidades acotan su efecto.
6. **Edición 2022 transcrita.** Se valida contra subtotales del informe, pero no proviene de texto extraído automáticamente.
7. **Validación del modelo.** Mide cómo predice la siguiente revisión de proyectos que ya están en la cartera, no el comportamiento de proyectos nuevos. Su desempeño cambia de un año a otro.
8. **Precio del cobre.** Agregado después de ver los datos; es una variable común a todos los proyectos de una edición y puede capturar tendencias temporales.
