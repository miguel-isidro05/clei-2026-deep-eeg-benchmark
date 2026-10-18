# Auditoría científica y de código V12

Fecha: 2026-10-16
Rama auditada: `exp/peterson-model-search-v12`
Base auditada: `f31add3c4f4a3b0d807436ee442880c87c2af601`

## Dictamen

V10 ya constituye un benchmark Peterson completo y V11 demuestra que `overlap` debe mantenerse
como condición principal within-session. V12 todavía no demuestra un modelo nuevo superior a
FBCNet: Exp02 obtuvo `+1.104` puntos porcentuales, con IC95% `[-1.068, 2.849]`, `p=.4375` y 4/6
victorias, pero el candidato era esencialmente FBCNet bajo otra receta.

La ruta principal recomendada es un decodificador espectroespacial consciente de confiabilidad de
canal, evaluado bajo degradaciones representativas de hardware low-cost. Difusión queda fuera de
la ruta principal hasta que un control simple y reproducible demuestre beneficio.

## Hechos consolidados

- Peterson/MI-OpenBCI contiene una sola sesión por participante. `RUN1`–`RUN4` son corridas de MI
  versus reposo; `RUN0` contiene movimiento real y debe excluirse.
- No puede llamarse `cross-session` a leave-one-run-out en Peterson. Los términos válidos son
  within-session, cross-run o leave-one-run-out.
- El split se realiza por ensayo antes de crear ventanas; las predicciones se agregan por ensayo.
  La implementación V12 auditada respeta esta condición.
- El preprocesamiento y las estadísticas de calidad se ajustan con train. El holdout
  `S07,S04,S05,S06` está bloqueado durante discovery.
- En V10, FBCNet es el baseline principal within-subject. En V11, overlap mejora el conjunto de
  modelos incluso al igualar exposición y actualizaciones.
- Exp01 y Exp02 refutan la formulación difusiva/energética actual: sus numerosos colapsos no se
  corrigen justificadamente aumentando capacidad.

## Novedad: límites y hueco defendible

No son reclamos nuevos por sí solos: channel attention, dynamic spatial filtering, channel
dropout, adaptive channel mixing, difusión para augmentación, difusión de un paso, ERD/ERS loss,
domain adaptation, sparse-to-dense ni trust-gated augmentation.

El hueco defendible es más específico:

> Decodificación espectroespacial consciente de confiabilidad para MI-EEG low-cost, condicionada
> por medidas objetivas de calidad de canal y validada bajo pérdida y corrupción controlada de
> electrodos reales OpenBCI.

Nombre descriptivo provisional: **Reliability-Guided Spectro-Spatial Network**. No se adoptará un
nombre propio ni se reclamará novedad hasta superar Exp04 y Exp05.

La arquitectura candidata conserva el sesgo fisiológico de FBCNet:

1. banco de filtros mu/beta;
2. filtrado espacial por banda;
3. descriptores de calidad ajustados solo con train;
4. gate residual inicializado como identidad;
5. channel dropout antes del gate para que este observe la corrupción;
6. pooling de log-varianza;
7. evaluación limpia primaria y estrés low-cost secundario.

Si esta versión no mejora, los dos siguientes cambios deben probarse por separado, no juntos:

- pirámide temporal explícita con soportes múltiples;
- pooling robusto de potencia: winsorizado, median-of-means o Huber.

## Bloqueos encontrados y correcciones aplicadas

1. **Inicialización no determinista.** El modelo se construía antes de fijar la semilla. Ahora el
   entrenador recibe una factory, fija la semilla y recién construye el modelo.
2. **Semillas no emparejadas.** El FBCNet manual usaba `seed*10000` y V10 `seed*1000`. Ambos usan
   ahora la misma función de semilla.
3. **Variación entre semillas incorrecta.** `seed_std` mezclaba sujetos y semillas. Ahora calcula
   primero la media por seed.
4. **Paridad susceptible a cancelación.** La puerta ya no usa solo el delta firmado medio; también
   limita el error absoluto pareado y el máximo delta absoluto por sujeto.
5. **Conteo de parámetros.** Los FIR congelados se confundían con parámetros activos. El reporte
   separa entrenables y congelados; FBCNet exacto informa 4,994 activos.
6. **Gate incapaz de aprender channel dropout.** El dropout ocurría después del gate. Ahora la
   corrupción se aplica antes del gate, igual que en la prueba de estrés.
7. **Latencia sesgada.** El gate se calculaba dos veces por batch. La inferencia devuelve logits y
   pesos del gate en una sola pasada.
8. **Ranking incompleto.** La robustez bajo canales ausentes y los parámetros activos participan
   ahora antes de la latencia en los desempates.
9. **Decisión de receta manipulable.** La fase quality recalcula la firma y exige estado
   `selected`; una decisión con paridad fallida no puede iniciar la fase B.
10. **GPU dentro de la identidad científica.** La GPU se conserva como metadato de runtime, pero
    se excluye del fingerprint científico para permitir shards equivalentes en `cuda:0/cuda:1`.

Consecuencia: cualquier Exp03 producido con `f31add3` antes de estas correcciones sirve solo para
diagnóstico de tiempo/errores. No debe entrar al análisis científico y debe regenerarse.

## Árbol de decisión al recibir resultados

| Escenario | Decisión |
|---|---|
| Fase A no pasa paridad | Bloquear fase B; investigar entrenador/receta. |
| Ningún candidato mejora limpio ni estrés | Cerrar gating; probar pirámide temporal y pooling robusto en experimentos separados. |
| Mejora estrés pero no limpio | Conservar como contribución de robustez, no como “mejor clasificador”. |
| Mejora limpia ≥1 pp, 4/6 victorias, κ y cuartil conservados | Promover 2–4 candidatos diversos a Exp04. |
| Exp04 no reproduce la mejora con 5 seeds × 5 folds | Rechazar el modelo nuevo; publicar benchmark V10/V11. |
| Exp04 reproduce | Congelar código/receta y abrir una sola vez el holdout en Exp05. |
| Exp05 mantiene dirección y efecto | Sustentar superioridad en Peterson discovery+holdout, con límites de una sola sesión. |

El holdout tiene cuatro sujetos; el mínimo `p` bilateral exacto es 0.125. Se usará para efecto,
intervalo, dirección y consistencia, no como requisito artificial de `p<.05`.

## Difusión: ruta futura separada

La difusión MI genérica no es novedosa. Si se retoma, la hipótesis con mayor fundamento es una
difusión residual pequeña sobre un prior PySimMIBCI, condicionada por dispositivo/calidad y con un
gate fail-closed. Debe compararse contra real-only, perturbaciones simples, PySim solo, DDPM
convencional y DDPM+gate. La métrica central será el decoding externo y el harm rate, no FID de
imágenes ni correlación de señal.

Esta ruta no se mezclará con Exp03/Exp04. Solo se justifica después de congelar el decoder real.

## Evidencia externa principal

- Peterson et al., Heliyon 2020: <https://doi.org/10.1016/j.heliyon.2020.e03425>
- Data paper MI-OpenBCI: <https://doi.org/10.1016/j.dib.2022.108225>
- FBCNet: <https://arxiv.org/abs/2104.01233>
- Dynamic spatial filtering: <https://doi.org/10.1016/j.neuroimage.2022.118994>
- Leakage en estudios EEG: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11099244/>
- Trust-Gated Augmentation: <https://doi.org/10.1038/s41746-026-02778-0>
- PySimMIBCI: <https://doi.org/10.1109/TNSRE.2024.3417311>

## Criterio de cierre inmediato

Antes de ejecutar en Cayetano deben pasar la suite completa, Ruff, smoke de recipe y quality,
auditor portable y verificación de árbol limpio. Después de generar resultados, se revisarán
paridad, hashes, colapsos, efectos por sujeto, estabilidad por seed, estrés, parámetros y latencia
antes de decidir Exp04.
