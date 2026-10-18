# Auditoría de resultados Exp03 `f31add3`

Estado científico: `exploratory_not_baseline`
Cobertura de evidencia: 91% (39 de 43 afirmaciones respaldadas o derivadas; cuatro hipótesis permanecen abiertas)

## Veredicto

La ejecución está completa y sus archivos son internamente coherentes, pero no debe usarse como evidencia confirmatoria. La versión `f31add3` construía los modelos antes de fijar la semilla y aplicaba la compuerta de calidad antes del `channel_dropout`. Por ello, la inicialización dependía del historial del proceso y la compuerta no veía durante el entrenamiento los canales anulados de forma sintética.[1][2]

El resultado útil es una hipótesis para la siguiente ola. Un FBCNet reducido de 16 filtros obtuvo la mayor precisión limpia en la búsqueda, mientras que entrenar con `channel_dropout=0.1` mejoró la tolerancia a uno o dos canales ausentes a costa de precisión limpia.[3][4] La compuerta G1/G2 no mostró una ventaja limpia consistente en esta ejecución, pero el orden incorrecto entre compuerta y dropout impide tratar ese resultado como una refutación de la idea.[2][5]

## Integridad del paquete

- Los manifiestos firmados contienen 186 celdas de receta y 486 celdas de calidad.[6]
- Los dos manifiestos, `recipe_decision.json` y `promotion_decision.json` conservan firmas SHA-256 válidas; `run_complete.json` referencia esas decisiones.[7]
- Los registros terminan con 93/93 celdas por GPU en receta y 243/243 por GPU en calidad, sin fallos del proceso.[8]
- Hubo cuatro colapsos de clase en la fase de calidad, todos en configuraciones con 48 filtros, `stride=8` y dropout de 0.1 o 0.2.[9]

## Resultados observados

La receta seleccionada fue `R3-best-wd1e-4`: 300 épocas, checkpoint por menor pérdida de validación y `weight_decay=1e-4`. Alcanzó accuracy media 0.8322, kappa 0.6644 y desviación entre las cinco medias por semilla de 0.0071.[10] La corrección posterior de `seed_std` no cambia la receta seleccionada.[11]

Los baselines de la fase de receta fueron CSP+LDA con accuracy 0.7502 y FBCNet-V10 con 0.8196.[12] La comparación de paridad entre el FBCNet manual en último checkpoint y FBCNet-V10 pasó los límites reforzados: diferencia media de -0.0085, diferencia absoluta pareada media de 0.0154 y máximo absoluto por sujeto de 0.0138.[13]

En la búsqueda de calidad, ejecutada solo con seed 0, `exp03-qcfg-002` obtuvo la mayor precisión limpia: accuracy 0.8485 y kappa 0.6969. Usa G0, sin dropout, 16 filtros espaciales y `stride=4`.[14] Esta configuración superó a cinco de seis sujetos frente al control predeclarado, pero la comparación está afectada por la inicialización no determinista de `f31add3`.[1][14]

La mejor configuración sencilla para canales ausentes fue `exp03-qcfg-017`, también G0, con `channel_dropout=0.1`, 48 filtros y `stride=4`. Su accuracy limpia fue 0.7696; con un canal ausente obtuvo 0.7572 y con dos, 0.7426.[15] Frente a `qcfg-002`, mejoró la condición de un canal ausente en cinco de seis sujetos y la de dos canales en los seis, aunque perdió 0.0789 de accuracy limpia.[15]

Promediando todas las configuraciones, `channel_dropout=0.1` redujo la accuracy limpia en 0.0459 frente a dropout 0.0. A la vez, elevó la accuracy media bajo canales ausentes de aproximadamente 0.679/0.632 a 0.753/0.729 para uno/dos canales en G0.[16] Los contrastes limpios entre compuertas no sobrevivieron corrección de Holm; G1 menos G0 fue -0.0040 y G2 menos G1 fue 0.0017.[17]

## Qué queda aceptado y qué queda abierto

Se conserva como señal de descubrimiento que el tamaño de 16 filtros merece confirmación y que el dropout de canales crea un frente precisión-resiliencia. No se acepta todavía que `qcfg-002` supere FBCNet-V10, que G1/G2 carezcan de utilidad ni que `qcfg-017` sea el modelo final. La fase de calidad utilizó una sola semilla y la versión ejecutada no controló la inicialización antes de construir cada modelo.[1][18]

El siguiente paso correcto es repetir Exp03 con la inicialización determinista, dropout antes de la compuerta, firmas de decisiones verificadas y contrastes factoriales de precisión limpia y estrés. Después de esa réplica se debe confirmar con cinco semillas un candidato limpio, uno resiliente y el control. Una nueva arquitectura solo se justifica si esa confirmación mantiene el frente observado.

## Diseño recomendado para la ola siguiente

La ruta principal propuesta es un único FBCNet de 16 filtros con entrenamiento mixto. Cada minibatch conservaría ejemplos limpios y añadiría una copia con uno o dos canales anulados. La función de pérdida combinaría clasificación limpia, clasificación corrupta y consistencia entre ambas predicciones. El objetivo sería recuperar parte de la mejora bajo canales ausentes sin perder los 7.9 puntos observados entre `qcfg-002` y `qcfg-017`.[inferencia a partir de 3,4,15,16]

La alternativa de menor riesgo es un sistema de dos expertos: `qcfg-002` para señales completas y `qcfg-017` cuando un detector determinista identifica canales ausentes. Esta ruta preservaría los dos extremos observados, pero exigiría dos checkpoints y debe reportarse como enrutamiento, no como una sola red.[inferencia a partir de 14,15]

La compuerta aprendida queda como ablación secundaria. Solo puede volver a evaluarse con dropout aplicado antes de calcular descriptores y con una pérdida que obligue a mantener desempeño limpio. No conviene ampliar G1/G2 antes de repetir el control corregido.[inferencia a partir de 1,2,5,17]

## Mapa de fuentes

1. Código `f31add3`, `runner.py` líneas 320-337 y 592-613; `training.py` líneas 84-105: el modelo se instancia antes de `set_seeds(seed)`.
2. Código `f31add3`, `models.py` líneas 225-236: `apply_quality_gate(x)` precede a `channel_dropout(gated)`.
3. `analysis/quality-readout.json`, entradas `exp03-qcfg-002` y `exp03-qcfg-017`.
4. `tables/quality-channel-stress.csv`, filas de `qcfg-002` y `qcfg-017`.
5. Reanálisis con el código corregido, `factorial_contrasts`: G1-G0 y G2-G1.
6. `manifests/recipe-manifest.json` y `manifests/quality-manifest.json`, longitudes de `cells` 186 y 486.
7. `decisions/*.json` y `run_complete.json`; firmas recalculadas con JSON canónico.
8. `logs/recipe-shard-{0,1}.log` y `logs/quality-shard-{0,1}.log`.
9. `quality_cells`: `qcfg-027/S08`, `qcfg-045/S10`, `qcfg-054/S08` y `qcfg-081/S08` contienen `validation_class_collapse`.
10. `analysis/recipe-readout.json`, `recipe_decision.selected_recipe` y `selection_summary`.
11. Reanálisis local con `_summary` corregido: medias por seed 0.8402, 0.8362, 0.8276, 0.8206 y 0.8364.
12. `analysis/recipe-readout.json`, `baseline_summary`.
13. Reanálisis local con `_parity_report` corregido.
14. `tables/quality-ranking.csv`, primera fila, y `manifests/quality-catalog.json`, `qcfg-002`.
15. `tables/quality-subject-metrics.csv` y `tables/quality-channel-stress.csv`, comparación pareada `qcfg-017` frente a `qcfg-002`.
16. Reanálisis factorial de `quality-subject-metrics.csv` y `quality-channel-stress.csv`.
17. Reanálisis local, `factorial_contrasts`, con permutación por sujeto y corrección de Holm.
18. `manifests/quality-manifest.json`: todas las celdas de calidad usan seed 0.

## Registro de afirmaciones no resueltas

| Afirmación | Estado | Evidencia necesaria |
|---|---|---|
| `qcfg-002` supera FBCNet-V10 | No confirmada | Repetición determinista y cinco semillas |
| El dropout mixto conserva precisión limpia y resistencia | Hipótesis | Ola con pérdida limpia/corrupta y consistencia |
| La compuerta de calidad mejora canales ausentes | No evaluada correctamente | Dropout antes de la compuerta y confirmación pareada |
| El sistema de dos expertos es preferible a una sola red | No comparada | Misma partición, semillas, latencia y presupuesto paramétrico |
