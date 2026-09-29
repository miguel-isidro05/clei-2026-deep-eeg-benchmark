# Resumen de auditoria cientifica

## Dictamen

El pipeline del paper quedo listo para la corrida completa en la PC de Cayetano. Cubre los cinco
decoders deep solicitados, cinco semillas, `within_split`, `within_session` de cinco folds,
LOSO para MI-OpenBCI, leave-one-session-out para Zhou2020 y Tavakolan2017, Wilcoxon pareado con
Holm, intervalos de confianza, efecto rank-biserial, control de calidad de senal, perfilado de
latencia, figuras y bloque LaTeX generados desde artefactos validados.

Los bloqueantes previos quedan resueltos en codigo y documentacion. BNCI2014_001 no forma parte
del perfil confirmatorio porque no contiene la clase `rest`; reducirlo a left-versus-right
responderia otra pregunta. La decision congelada es mantener MI-OpenBCI como dataset low-cost
primario y usar Zhou2020/Tavakolan2017 como validacion externa task-matched.

Lo que sigue pendiente no es de implementacion sino de ejecucion: aun no existen resultados
completos de las 3.600 celdas. Por eso no debe declararse ganador, mejora significativa ni ranking
final hasta que `scripts/run_cayetano.sh` termine y `scripts/run_statistics.py`,
`scripts/generate_figures.py` y `scripts/generate_manuscript_results.py` pasen el gate
confirmatorio.

## Evidencia de ejecucion local

- `python scripts/run_paper.py --phase all --plan-only` produjo `planned_cells=3600`.
- El perfil confirmatorio rechaza cualquier numero de epocas distinto de 300.
- La inferencia confirmatoria se bloquea si faltan celdas, seeds, manifiestos o si los resultados
  no coinciden con el fingerprint esperado.
- Las ejecuciones incompletas solo pueden generar salidas descriptivas marcadas como exploratorias.
- `scripts/run_cayetano.sh` deja consola completa en `results/logs/paper-*.log` y resumen en
  `results/EXPERIMENT_LOG.md`.

La matriz de decisiones y trazabilidad esta en `SCIENTIFIC_AUDIT.md`, `RESOLUTION_REPORT.md`,
`COMPARABILITY_REPORT.md`, `SCIENTIFIC_CHANGELOG.md` y `PATCHES.md`.
