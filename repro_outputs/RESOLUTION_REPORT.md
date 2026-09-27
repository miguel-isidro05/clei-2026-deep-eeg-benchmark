# Resolución de bloqueantes científicos

Fecha: 2026-09-27

## Diseño corregido

| Bloqueante | Resolución | Evidencia |
|---|---|---|
| 5-fold ausente | `within_session` se añadió al bloque primario | `src/deepbench/paper_profile.py` y 250 celdas planificadas |
| BNCI2014_001 no task-matched | El perfil usa Zhou2020 y Tavakolan2017, ambos `right_hand` versus `rest` y multisesión | `docs/MOABB_DATASET_INVENTORY.md` |
| Procedencia de hiperparámetros | Cada receta registra `provenance_id` y política sin test tuning | `docs/HYPERPARAMETER_PROVENANCE.md` |
| Identidad de entorno/datos | El fingerprint incluye versiones científicas y SHA-256 de los arrays estandarizados | `src/deepbench/runner.py`, smoke real |
| Tamaño de efecto | Cada contraste añade correlación rank-biserial pareada | `src/deepbench/statistics.py` |
| Contabilidad de muestras | Se genera `sample_accounting.csv` con trials, ejemplos y clases por fold | `src/deepbench/statistics.py` |
| TeX histórico incompatible | Se creó `manuscript/revision_v2` sin CSP, DeepConvNet, Friedman, bootstrap ni números antiguos | `manuscript/revision_v2/paper_CLEI2026.tex` |

## Inventario MOABB

La instalación fijada de MOABB 1.5.0 contiene 54 datasets con etiqueta imagery y 25 multisesión.
El filtro `right_hand` + `rest` + multisesión devuelve exactamente Zhou2020, Tavakolan2017 y
Stieger2021. Stieger2021 queda fuera por su costo, tamaño y licencia no comercial.

## Evidencia de ejecución

- 38 pruebas aprobadas.
- Ruff aprobado.
- Preflight de las cinco arquitecturas aprobado.
- Plan paper: 3.350 celdas, incluidas 250 `within_session` y 800 `cross_session`.
- Smoke real MI-OpenBCI/S02/EEGNet aprobado con hashes de entorno y datos, procedencia de receta,
  conteos de clase y `sample_accounting.csv`.
- El smoke incompleto no generó inferencia confirmatoria.

## Pendiente legítimo

No existen todavía resultados completos. `generated_results.tex` contiene un aviso explícito y no
debe sustituirse hasta que la auditoría de celdas y seeds termine sin faltantes. La compilación TeX
no pudo probarse en esta máquina porque no hay un motor LaTeX instalado.
