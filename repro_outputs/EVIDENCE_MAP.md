# Mapa de evidencia

> **Auditoría histórica de `e39fd1e`.** Los conteos, datasets y rutas siguientes documentan el
> estado auditado antes de las correcciones. Consulte `RESOLUTION_REPORT.md` para la evidencia
> vigente y no use este archivo como checklist de ejecución.

## Feedback canónico

- `../Reviews/Retroalimentacion_consolidada_NeurIPS_2026.xlsx`, hoja `Ejes consolidados`, filas
  A01-A35: requisitos consolidados, prioridad, acción y evidencia de cierre.
- `../Reviews/NEURIPS_2026_OpenReview_decision_feedback_latest.txt`: decisión y comentarios
  textuales de los dos revisores.
- `../Reviews/README_feedback_index.md`: establece el workbook como tracker principal y el TXT
  como fuente más reciente.

## Diseño y alcance declarados

- `README.md:1-134`: modelos, datasets, ejecución, salidas y definición de latencia.
- `docs/EXPERIMENT_SPEC.md:1-94`: protocolos, preprocessing, semillas, unidad estadística y
  criterio de cierre.
- `docs/FEEDBACK_COVERAGE.md:1-24`: mapa previo de feedback, solución y límites.
- `docs/MOABB_DATASET_DECISION.md:1-27`: razón de uso de MOABB y rol de cada dataset.

## Evidencia de código

- `src/deepbench/config.py:9-100`: modelos, seeds, sujetos, tareas y datasets.
- `src/deepbench/models.py:20-139`: recetas, arquitecturas y entrenamiento fijo.
- `src/deepbench/datasets.py:64-180`: adaptadores MI-OpenBCI y MOABB.
- `src/deepbench/preprocessing.py:23-179`: filtros, ICA, normalización y ventanas.
- `src/deepbench/evaluation.py:24-348`: splits, seeds, predicciones held-out e historial.
- `src/deepbench/runner.py:21-209`: fingerprint, reanudación y escritura atómica.
- `src/deepbench/statistics.py:92-302`: CI t, Wilcoxon, Holm, agregación de seeds y gate de
  completitud.
- `src/deepbench/latency.py:14-87`: entorno, sincronización y alcance de latencia.
- `scripts/run_paper.py:16-148`: matriz experimental realmente planificada.
- `scripts/run_statistics.py:16-48`: auditoría de celdas esperadas antes de inferencia.

## Evidencia de pruebas y ejecución

- `tests/`: 26 pruebas aprobadas el 2026-09-27.
- `ruff check .`: aprobado el 2026-09-27.
- `scripts/preflight.py`: aprobado para las cinco arquitecturas con PyTorch 2.12.1,
  Braindecode 1.5.2, MOABB 1.5.0 y MNE 1.12.1.
- Plan-only: 2,950 celdas; 0 `within_session`, 725 `cross_session`, 250 `loso`.
- Smoke: MI-OpenBCI/S02/EEGNet/seed0/una época, 105 trials train y 45 test.
- Gate negativo: `run_statistics.py` terminó con error ante 2,950 celdas faltantes.
- Gate exploratorio: un smoke incompleto produjo `INCOMPLETE_EXPLORATORY_ONLY.txt` y ninguna
  tabla Wilcoxon-Holm.

## Manuscrito no compatible

- `../Paper_final/PAPER_CLEI_V2_COPY/paper_CLEI2026.tex:48`: abstract con CSP y cuatro redes.
- `paper_CLEI2026.tex:148`: seed fija 0 y una nota interna sin resolver.
- `paper_CLEI2026.tex:220-263`: resultados históricos con bootstrap y Friedman.
- `paper_CLEI2026.tex:443`: reconoce una sola seed y soporte temporal desigual.
- `paper_CLEI2026.tex:453`: promete liberar código en el futuro.

## Afirmaciones no verificadas

- [UNSOURCED] Convergencia de todos los modelos a 300 epochs. Requiere las historias completas.
- [UNSOURCED] Superioridad de cualquier modelo. Requiere Wilcoxon-Holm completo.
- [UNSOURCED] Beneficio multisemilla de overlap/nonoverlap. Requiere la ablación completa.
- [UNSOURCED] Robustez cross-session. Requiere Zhou2020 y BNCI2014_001 completos.
- [UNSOURCED] Identidad ocular o muscular de los IC rechazados. Kurtosis no la demuestra.
- [UNSOURCED] Readiness online, clínica o doméstica. El estudio es offline y la latencia es
  model-only.
