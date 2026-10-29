# Deriva de la cartera de inversión minera en Chile (2014–2025)

¿Cuánto se atrasan los proyectos mineros que anuncia la cartera de Cochilco y cuáles llegan a operar? Este repositorio sigue a {{n_proyectos|ent}} proyectos a lo largo de {{n_ediciones|ent}} ediciones anuales del catastro de inversión minera (2014–2025) y mide su deriva de plazo y de capex y su desenlace.

*English summary below.*

![Página 1 del tablero: destino de cada proyecto](docs/img/c3_p1_cartera.png)

## Hallazgos principales

1. **La condición que asigna Cochilco separa a los proyectos que llegan a operar.** Entre los proyectos que entraron al catastro desde 2015 (n = {{n_cohorte_incidente|ent}}), a 6 años entró en operación el {{cif_op_6_base_probable|prop0}} de los Base o Probable (IC 95 %: {{ic_op_inf_6_base_probable|prop0}}–{{ic_op_sup_6_base_probable|prop0}}) frente al {{cif_op_6_posible_potencial|prop0}} de los Posible o Potencial ({{ic_op_inf_6_posible_potencial|prop0}}–{{ic_op_sup_6_posible_potencial|prop0}}). La brecha se mantiene en los 7 escenarios de sensibilidad.
2. **La fecha anunciada se corre.** El {{pct_transiciones_atraso|pct1}} de las {{n_transiciones|ent}} revisiones entre ediciones atrasa la puesta en marcha. A los 4 años de entrar al catastro, el proyecto mediano que sigue en él acumula {{mediana_deriva_anio4|ent}} años de atraso. Solo el {{pct_fiabilidad_a_tiempo|pct0}} de los proyectos que operaron lo hizo en el año anunciado al entrar.
3. **El modelo ordena el riesgo de atraso mejor que el azar, pero no en todos los años.** Con validación temporal ({{n_cortes|ent}} cortes), el AUC agregado es {{auc_agregado_completo|d2}} (IC 95 %: {{auc_agregado_completo_ic_inf|d2}}–{{auc_agregado_completo_ic_sup|d2}}) y varía entre {{auc_corte_min|d2}} y {{auc_corte_max|d2}} según el año. La mejora en probabilidades frente a la tasa base no es concluyente: su intervalo incluye el cero.

Todas las cifras de este repositorio provienen de un registro de {{n_cifras_verificadas|ent}} cifras recalculadas por código independiente ([`docs/registro_cifras.csv`](docs/registro_cifras.csv)). Los resultados son asociativos, no causales.

## Documentos

- [Brief](docs/brief.md): el estudio en una línea, en lenguaje simple y en su capa técnica.
- [Método](docs/metodo.md): fuentes, identidad de proyectos, conciliación con Cochilco, modelos, validación y limitaciones.
- [Preguntas difíciles](docs/preguntas_dificiles.md): las objeciones más fuertes al estudio y lo que el repositorio permite responder.

## Tablero

Cuatro páginas en Power BI: cartera, deriva, desenlaces y confiabilidad.

| | |
|---|---|
| ![Cartera](docs/img/c3_p1_cartera.png) | ![Deriva](docs/img/c3_p2_deriva.png) |
| ![Desenlaces](docs/img/c3_p3_desenlaces.png) | ![Confiabilidad](docs/img/c3_p4_confiabilidad.png) |

## Cómo reproducir

Requiere Python 3.12 y [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m cpdrift.fuentes      # descarga las fuentes y verifica su SHA-256
uv run python -m cpdrift              # pipeline completo y registro verificado
uv run python -m cpdrift.publicacion  # regenera README y documentos desde el registro
uv run pytest -q
```

Los informes de Cochilco se descargan desde sus URL públicas, listadas en [`data/reference/fuentes_raw.csv`](data/reference/fuentes_raw.csv). La planilla de precios del Banco Mundial (Pink Sheet) se actualiza cada mes: si su hash difiere, el pipeline avisa y continúa.

## Estructura

| Ruta | Contenido |
|---|---|
| `src/cpdrift/` | Extracción, identidad, conciliación, indicadores, modelos, verificación y publicación |
| `data/reference/` | Tablas curadas con cita a página: transcripción 2022, egresos declarados, cruce manual, ajustes, regiones y manifiesto de fuentes |
| `docs/` | Brief, método, preguntas difíciles, registro de cifras y capturas del tablero |
| `tests/` | Pruebas unitarias y de higiene |

## Fuentes y licencia

Datos: Comisión Chilena del Cobre (Cochilco), *Inversión en la minería chilena — Cartera de proyectos*, ediciones 2014 a 2025; Banco Mundial, *Commodity Price Data (Pink Sheet)*; U.S. Bureau of Labor Statistics, CPI-U. Los documentos fuente no se redistribuyen. El código se publica bajo licencia MIT (ver [`LICENSE`](LICENSE)).

---

## English summary

This repository follows {{n_proyectos|ent}} mining projects across {{n_ediciones|ent}} annual editions (2014–2025) of the investment portfolio published by Cochilco, Chile's copper commission, and measures schedule drift, capex drift and outcomes.

- **Cochilco's assigned condition separates projects that reach operation.** For projects entering the portfolio from 2015 onwards (n = {{n_cohorte_incidente|ent}}), {{cif_op_6_base_probable|prop0}} of Base or Probable projects were in operation after 6 years (95% CI {{ic_op_inf_6_base_probable|prop0}}–{{ic_op_sup_6_base_probable|prop0}}), against {{cif_op_6_posible_potencial|prop0}} of Possible or Potential projects ({{ic_op_inf_6_posible_potencial|prop0}}–{{ic_op_sup_6_posible_potencial|prop0}}). The gap holds in all 7 sensitivity scenarios.
- **Announced start-up dates slip.** {{pct_transiciones_atraso|pct1}} of {{n_transiciones|ent}} edition-to-edition revisions delay the start-up year. Four years after entry, the median project still in the portfolio has accumulated {{mediana_deriva_anio4|ent}} years of delay.
- **A temporally validated model ranks delay risk better than chance, but not every year:** pooled AUC {{auc_agregado_completo|d2}} (95% CI {{auc_agregado_completo_ic_inf|d2}}–{{auc_agregado_completo_ic_sup|d2}}), {{auc_corte_min|d2}} to {{auc_corte_max|d2}} across {{n_cortes|ent}} cuts. The improvement in probability accuracy over the base rate is not conclusive.

Every figure comes from a register of {{n_cifras_verificadas|ent}} independently recalculated figures. Results are associative, not causal. Documentation is in Spanish.
