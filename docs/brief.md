<!-- Generado desde docs/plantillas con cpdrift.publicacion. No editar a mano. -->
# Brief · Deriva de la cartera de inversión minera en Chile (2014–2025)

## En una línea
Los proyectos que Cochilco clasifica como Base o Probable llegan a operar mucho más que los Posible o Potencial, y la fecha anunciada se posterga en la mitad de las revisiones anuales.

## En lenguaje simple
Cada año, Cochilco publica la lista de grandes proyectos mineros que se esperan en Chile, con la fecha en que deberían empezar a operar, cuánto costarían y qué tan probable es que se concreten. Este estudio siguió a 133 de esos proyectos durante 12 ediciones de la lista para ver qué pasó con ellos.

- **La etiqueta de Cochilco sirve.** De cada 10 proyectos que entraron a la lista como "Base" o "Probable", unos 7 estaban operando 6 años después. De los "Posible" o "Potencial", solo unos 2.
- **Las fechas se corren.** En la mitad de las actualizaciones anuales (50 %), la fecha de inicio se posterga. Cuatro años después de aparecer, el proyecto típico que sigue en la lista lleva 3 años de atraso.
- **Se puede anticipar quién se atrasa, pero no con precisión.** Un modelo entrenado con años anteriores ordena qué proyectos tienen más riesgo de atrasarse mejor que el azar, aunque no todos los años y sin dar probabilidades confiables.

Ninguno de estos resultados dice por qué ocurre. Son patrones, no causas.

## Capa técnica
- **Datos:** 12 ediciones de la cartera de Cochilco, 133 proyectos, 409 transiciones entre ediciones. Edición 2022 transcrita y validada contra los subtotales del informe.
- **Desenlaces:** declarados por Cochilco y conciliados con su cuadro de cambios en 11 pares de ediciones (28 combinaciones exactas y 5 con ajuste documentado).
- **Riesgos competitivos (Aalen–Johansen, cohorte incidente n = 64):** operación a 6 años 73 % (IC 95 % 53 %–92 %) en Base o Probable frente a 23 % (5 %–46 %) en Posible o Potencial.
- **Deriva (OLS, errores agrupados por proyecto):** la condición Potencial se asocia con 1,16 años más de atraso por año transcurrido que Base; rango entre sensibilidades 1,05–1,19.
- **Predicción (logit de Firth, validación de origen móvil, 7 cortes):** AUC 0,66 (IC 95 % 0,59–0,73); por corte 0,50–0,82. Mejora del Brier no concluyente.
- **Rigor:** 111 cifras con recálculo independiente; 7 escenarios de sensibilidad; fuentes verificadas por SHA-256.
- **Principal limitación:** 18 egresos se identifican por cuadre de capex y no por nombre; en el escenario más pesimista, la operación a 6 años de Base o Probable baja a 52 %.

Detalle: [método](metodo.md) · [preguntas difíciles](preguntas_dificiles.md) · [repositorio](../README.md).
