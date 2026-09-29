# Parches aplicados

- Rama: `fix/scientific-blockers`
- Riesgo máximo: medio, porque se corrigió el significado del contraste de augmentación antes de
  producir resultados confirmatorios.
- Fidelidad al objetivo: aclarada. Se conservan los cinco modelos deep, datasets, particiones,
  métricas, semillas y estadística; se iguala el presupuesto de cómputo de la ablación.

## Cambios verificados

- Commit base `3c859caa182925ddb22645f0500fdd5c722d23f8`: entorno limitado a `conda-forge`.
- Commit `f06484c870ba1fed85ffa7ebffb4004aeb5454cc`: bloqueantes científicos, auditoría, calidad de
  señal, figuras y ejecución reanudable.
- Revisión final posterior: manifiesto de estadísticas, figuras completas, lanzador robusto para
  dos GPU y generación automática del bloque LaTeX de resultados.

Verificación: 60 pruebas, Ruff, preflight local, plan de 3.600 celdas, rechazo de épocas no
confirmatorias y sintaxis Bash. Los resultados científicos continúan pendientes de ejecución.

La revisión independiente final añadió cuatro cierres: hash del código científico en celdas y
manifiestos, hashes de CSV/figuras/latencia, tabla TeX de Wilcoxon-Holm con IC y rank-biserial, y
pruebas negativas contra artefactos alterados.

La segunda pasada de revisión cerró además: decisión explícita del autor para el conjunto de cinco
decoders deep, selección `within_session` cinco-fold como inferencia primaria del TeX, matriz de
confusión primaria basada en `within_session` e incluida en LaTeX, y preflight estricto de dos GPU
para `scripts/run_cayetano.sh`.
