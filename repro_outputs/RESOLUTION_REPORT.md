# Resolución de bloqueantes científicos

Fecha: 2026-09-27

## Diseño corregido

| Bloqueante | Resolución | Evidencia |
|---|---|---|
| 5-fold ausente | `within_session` se añadió al bloque primario | `src/deepbench/paper_profile.py` y 250 celdas planificadas |
| BNCI2014_001 no task-matched | El perfil usa Zhou2020 y Tavakolan2017, ambos `right_hand` versus `rest` y multisesión | `docs/MOABB_DATASET_INVENTORY.md` |
| Procedencia de hiperparámetros | Cada receta registra `provenance_id` y política sin test tuning | `docs/HYPERPARAMETER_PROVENANCE.md` |
| Identidad de entorno/datos | El fingerprint incluye versiones científicas, origen/commit de dependencias VCS y SHA-256 de los arrays estandarizados | `src/deepbench/runner.py`, smoke real |
| Tamaño de efecto | Cada contraste añade correlación rank-biserial pareada | `src/deepbench/statistics.py` |
| Contabilidad de muestras | Se genera `sample_accounting.csv` con trials, ejemplos y clases por fold | `src/deepbench/statistics.py` |
| TeX histórico incompatible | Se creó `manuscript/revision_v2` sin CSP, DeepConvNet, Friedman, bootstrap ni números antiguos | `manuscript/revision_v2/paper_CLEI2026.tex` |

## Inventario MOABB

La instalación fijada de MOABB 1.5.0 contiene 54 datasets con etiqueta imagery y 25 multisesión.
El filtro `right_hand` + `rest` + multisesión devuelve exactamente Zhou2020, Tavakolan2017 y
Stieger2021. Stieger2021 queda fuera por su costo, tamaño y licencia no comercial.

## Evidencia de ejecución

- 47 pruebas aprobadas.
- Ruff aprobado.
- Preflight de las cinco arquitecturas y del lector BCI2000 aprobado.
- Plan paper: 3.350 celdas, incluidas 250 `within_session` y 800 `cross_session`.
- Smoke real MI-OpenBCI/S02/EEGNet aprobado con hashes de entorno y datos, procedencia de receta,
  conteos de clase y `sample_accounting.csv`.
- Smoke real Tavakolan2017/sujeto 1/EEGNet aprobado: 160 trials balanceados, 32 canales,
  384 muestras, cuatro sesiones de 40 trials y cuatro folds `within_split` completados.
- El smoke incompleto no generó inferencia confirmatoria.
- La revisión independiente de Python detectó y se corrigieron tres riesgos: dependencia opcional
  ausente para Tavakolan2017, métricas no finitas antes de Wilcoxon-Holm y commits solo
  documentales que invalidaban la reanudación.
- La revisión científica independiente aprobó el diseño bajo el alcance congelado `right_hand`
  frente a `rest`; corrigió la descripción del perfil externo y acotó el paralelismo a hardware y
  entornos homogéneos.
- La revisión final detectó que el análisis confirmatorio podía fallar abierto sin un manifiesto
  completo. El gate ahora exige cobertura de las tres fases y la igualdad exacta entre celdas
  esperadas y presentes; cualquier ausencia o celda extra bloquea Wilcoxon-Holm.
- La duración se documentó por dataset: cuatro segundos para MI-OpenBCI/Zhou2020 y tres para
  Tavakolan2017, idéntica entre modelos dentro de cada dataset.

## Pendiente legítimo

No existen todavía resultados completos. `generated_results.tex` contiene un aviso explícito y no
debe sustituirse hasta que la auditoría de celdas y seeds termine sin faltantes. La compilación TeX
no pudo probarse en esta máquina porque no hay un motor LaTeX instalado.
