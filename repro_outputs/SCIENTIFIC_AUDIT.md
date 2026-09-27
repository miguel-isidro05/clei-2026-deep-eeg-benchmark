# Auditoría científica del benchmark

> Auditoría histórica de `e39fd1e`. Consulte `RESOLUTION_REPORT.md` para el estado posterior de los
> bloqueantes. Este archivo no se reescribe para preservar la trazabilidad del dictamen original.

Fecha: 2026-09-27

Revisión auditada: `e39fd1eb8a4c2ce06d4f9267063633379afc54e8`

Alcance: código, tests, documentación, tracker de 35 ejes y manuscrito TeX activo.

## Escala

- **CUMPLE**: la solución está implementada y tiene evidencia estructural o de prueba.
- **PARCIAL**: existe una solución útil, pero falta una decisión, salida o documentación necesaria.
- **FALTA**: el requisito no está implementado o contradice el artefacto actual.
- **PENDIENTE DE EJECUCIÓN**: el código está preparado, pero no existen resultados completos.
- **NO APLICA**: el cambio de alcance elimina el contraste original y debe explicarse al revisor.

## Hallazgos que deben resolverse antes de la corrida completa

### P0. El perfil paper omite el 5-fold within-session

`_protocol_splits` implementa cinco folds estratificados, pero `run_paper.py` no los incluye en
ninguna fase. El manifiesto planificado confirmó 0 celdas `within_session`. Esto contradice A09 y el
manuscrito, que todavía presenta dicho protocolo. Hay que incluirlo para MI-OpenBCI o retirar y
justificar explícitamente su exclusión antes de ver resultados.

Evidencia: `src/deepbench/evaluation.py:51-61`, `scripts/run_paper.py:70-96` y ejecución plan-only.

### P0. La definición de BNCI2014_001 no coincide con el tracker canónico

El tracker A01/E03 pide el protocolo estándar de cuatro clases. El código configura únicamente
left-hand versus right-hand y todos los modelos tienen dos salidas. Si se conserva el diseño
binario, debe registrarse como desviación preespecificada motivada por comparabilidad y no como
reproducción del benchmark 2a de cuatro clases.

Evidencia: `src/deepbench/config.py:86-91`, `src/deepbench/models.py:96-102` y
`Retroalimentacion_consolidada_NeurIPS_2026.xlsx`, hoja `Plan experimental`, fila E03.

### P0. Recetas y compatibilidad aún no tienen trazabilidad suficiente

Las recetas están fijadas en código y no usan el test para seleccionar épocas, pero falta una tabla
que vincule cada valor con su fuente o con una decisión preespecificada. Además, el fingerprint de
celda cubre código, receta, epochs y hardware, pero no versiones de paquetes ni identidad/versiones
de los datos. Un cambio de entorno o de archivos de datos con el mismo código puede reutilizar el
mismo fold cache.

Evidencia: `src/deepbench/models.py:20-35`, `src/deepbench/runner.py:55-89`,
`src/deepbench/reproducibility.py:53-88`.

## Matriz de los 35 ejes canónicos

| ID | Veredicto | Evidencia y lectura | Acción mínima restante |
|---|---|---|---|
| A01 | PARCIAL + PENDIENTE DE EJECUCIÓN | Zhou2020 y BNCI2014_001 están en el perfil (`run_paper.py:87-96`), pero no hay resultados ni análisis modelo por dataset. | Ejecutar; reportar datasets por separado; no atribuir diferencias al hardware. |
| A02 | FALTA | No se incluye un segundo dataset low-cost. Zhou2020 y BNCI2014_001 son validaciones externas research-grade. | Añadir uno compatible o limitar todos los claims a MI-OpenBCI. |
| A03 | PARCIAL + PENDIENTE DE EJECUCIÓN | Wilcoxon bilateral, Holm y CI pareados están implementados (`statistics.py:174-217`); falta tamaño de efecto. | Añadir rank-biserial u otro efecto pareado y ejecutar la matriz completa. |
| A04 | CUMPLE EN CÓDIGO + PENDIENTE DE EJECUCIÓN | Seeds 0-4 (`config.py:16`), split fijo y seed ICA fija por fold (`evaluation.py:194-197`); se promedian dentro de sujeto (`statistics.py:120-123`). | Completar cinco seeds y publicar la dispersión. |
| A05 | CUMPLE BAJO EL NUEVO ALCANCE | CSP fue retirado. Los cinco modelos deep reciben trials completos en el primario y condiciones idénticas en la ablación (`run_paper.py:70-84`). | Explicar que el confusor clásico-versus-deep dejó de formar parte de la pregunta. |
| A06 | NO APLICA A CSP; CUMPLE PARA DL | La matriz deep incluye `center`, `nonoverlap` y `overlap` para los cinco modelos y registra trials/ejemplos (`evaluation.py:166-169`). | Reescribir la respuesta al revisor; no reutilizar la tabla CSP histórica. |
| A07 | CUMPLE EN DISEÑO + PENDIENTE DE EJECUCIÓN | Split antes de ventanas, test a una predicción por trial y control no solapado (`preprocessing.py:154-178`). | Ejecutar las tres condiciones y presentar deltas pareados por sujeto. |
| A08 | PARCIAL | Cada celda conserva `n_train_trials`, `n_train_examples` y `n_test_trials`; no existe una tabla automática de contabilidad. | Añadir generador CSV por dataset/protocolo/fold/clase/condición. |
| A09 | FALTA EN EL PERFIL PAPER | 70/30, 5-fold y LOSO existen en la API, pero el perfil ejecuta 70/30 y LOSO, no 5-fold (`run_paper.py:70-85`). | Incluir `within_session` para MI-OpenBCI o retirar el protocolo con justificación preespecificada. |
| A10 | CUMPLE | Z-score e ICA se ajustan con train; ventanas no cruzan trials; tests verifican splits y normalización (`tests/test_protocols.py:23-40`, `tests/test_preprocessing.py:8-34`). | Añadir una prueba específica del camino ICA real. |
| A11 | PARCIAL | Las recetas son fijas y no hay early stopping ni test tuning (`models.py:92-125`), pero no hay procedencia por hiperparámetro. | Crear tabla fuente/decisión/justificación por modelo antes de correr. |
| A12 | PARCIAL DELIBERADO | El primario usa `none`; la sensibilidad usa FastICA train-only y kurtosis mayor que 10, máximo dos, con log completo (`preprocessing.py:28-93`). No identifica ocular/muscular. | Mantener el claim limitado o usar un clasificador validado de IC/EOG si se requiere identificación fisiológica. |
| A13 | PARCIAL | La lista fija omite S01/S11 (`config.py:45-56`) y el smoke confirmó 150 trials en S02, pero no hay tabla completa ni razón upstream verificada. | Generar conteos por sujeto/clase y documentar la ausencia con la fuente del dataset. |
| A14 | PARCIAL | Clases exactas y versiones directas están fijadas (`models.py:13`, `pyproject.toml`), y el manifest registra versiones. No hay lock transitive ni release. | Generar lock, release/tag y apéndice de parámetros exactos. |
| A15 | PARCIAL | FBCNet usa seis subbandas dentro de 8-30 Hz (`models.py:50-57`), coherente con el preprocesamiento común, pero el documento metodológico no enumera ni justifica el cambio. | Documentar bandas y aclarar que difiere del esquema 4-40 del tracker histórico. |
| A16 | PARCIAL | El código está en GitHub, pero el cierre exige release verificable, licencia, splits/manifests, predicciones y checksums. | Publicar release versionada con licencia y artefactos finales. |
| A17 | CUMPLE EN DOCS; FALTA EN MANUSCRITO | README limita la comparación causal entre datasets; el TeX todavía usa resultados y alcance anteriores. | Reescribir abstract, título, discusión y conclusión después de los nuevos resultados. |
| A18 | CUMPLE | La especificación limita LOSO a transferencia dentro del dataset (`EXPERIMENT_SPEC.md:1-17`); el manuscrito histórico también lo reconoce. | Conservar el lenguaje en toda la revisión. |
| A19 | CUMPLE PARA LATENCIA DE MODELO | Se mide solo forward pass y se excluye adquisición/preprocesamiento (`latency.py:43-87`). | Llamarlo `model inference latency`; no afirmar readiness clínica u online. |
| A20 | PARCIAL | El repo es un benchmark de arquitecturas existentes, pero el manuscrito final aún no está alineado con EEGInceptionMI ni con la contribución revisada. | Reescribir contribuciones sin afirmar una arquitectura nueva. |
| A21 | CUMPLE | Incluye EEGConformer y EEGInceptionMI, además de EEGNet, FBCNet y ShallowConvNet (`config.py:9-15`). | Justificar familias representativas, no estado del arte exhaustivo. |
| A22 | FALTA COMO ANÁLISIS | Se guardan resultados por sujeto, pero no existe análisis preespecificado de calidad/no estacionariedad ni discusión nueva. | Interpretar heterogeneidad tras correr; separar evidencia de hipótesis. |
| A23 | PARCIAL | Se guardan scores y AUC por celda (`metrics.py:16-28`), pero no hay script ni regla de pooling para ROC. | Preferir AUC por sujeto con CI o documentar exactamente el pooling de cualquier figura. |
| A24 | PARCIAL EDITORIAL | Kappa se calcula por sujeto con sklearn; la definición del manuscrito debe revisarse. | Definirlo como acuerdo observado corregido por el esperado al azar. |
| A25 | FALTA AUDITORÍA FINAL | La bibliografía del TeX no fue reconciliada con el nuevo estudio. | Ejecutar auditoría de citas y DOI después de congelar la fuente revision_v2. |
| A26 | FALTA | El TeX conserva errores, notas internas y bloques históricos. | Copy-edit académico completo después de reemplazar resultados. |
| A27 | PENDIENTE DEL VENUE | Es un requisito de plantilla, no del pipeline. | Aplicar la guía del próximo venue al artefacto final. |
| A28 | PARCIAL | Se conserva `training_history` por epoch (`evaluation.py:139-169`), pero no hay figura ni criterio automático de diagnóstico. | Generar curvas suplementarias y revisar convergencia sin seleccionar epochs post-hoc. |
| A29 | PARCIAL | Se guardan `y_true`/`y_pred`, suficientes para matrices, pero no hay generador estandarizado. | Añadir figura normalizada por fila con N y regla de pooling. |
| A30 | CUMPLE EN CÓDIGO + PENDIENTE DE EJECUCIÓN | Leave-one-session-out está implementado y probado (`evaluation.py:30-37`, `tests/test_protocols.py:23-30`) para Zhou2020 y BNCI2014_001. | Completar ambas corridas y no extrapolar cross-session a MI-OpenBCI. |
| A31 | FALTA, BLOQUEANTE PARA EL PAPER | El TeX activo sigue describiendo otro benchmark y otros resultados. | Crear fuente `revision_v2`, preservar históricos y reemplazar solo con salidas verificadas. |
| A32 | CUMPLE EN CÓDIGO; PENDIENTE EN TABLAS | Todas las métricas se calculan sobre predicciones held-out (`evaluation.py:251-271`), no sobre train. | Etiquetar explícitamente held-out test en cada tabla y leyenda. |
| A33 | PARCIAL | Hay SD entre sujetos, SD entre seeds y CI t (`statistics.py:144-170`, `194-206`); falta tamaño de efecto. | Añadir efecto pareado y su convención de signo. |
| A34 | CUMPLE PARA EL ALCANCE DECLARADO | Los cinco modelos se miden con mismo backend, dispositivo, input, warm-up y batch 1/64 (`profile_latency.py:16-50`). No es end-to-end ni energía. | Ejecutar CPU y CUDA por separado si se comparan plataformas; no mezclar ambas en un ranking. |
| A35 | CUMPLE | MOABB se usa para datasets y Braindecode para los cinco decoders (`datasets.py:93-180`, `models.py:12-13`). | Conservar versiones y validar descarga en la PC de Cayetano. |

## Hallazgos transversales adicionales

1. **Resultados**: no existen resultados completos del benchmark nuevo. Ninguna frase de ganador,
   robustez o mejora por augmentación está cerrada.
2. **Licencia y disponibilidad**: el repositorio no contiene `LICENSE`, `CITATION.cff` ni una
   declaración formal de disponibilidad/licencia de cada dataset.
3. **Tests faltantes**: no hay prueba automatizada de ICA con convergencia real, adaptadores MOABB
   con datos reales, tabla de sample accounting, plots ni compatibilidad de dependencias/datos.
4. **Entorno actual de auditoría**: los scripts requirieron `PYTHONPATH=src` porque el entorno
   `eeg-diffusion` no tenía instalado este proyecto. El entorno nuevo creado con `environment.yml`
   sí debe instalarlo con `pip -e .`; conviene comprobarlo en la PC objetivo.
5. **Sharding**: el análisis exige el mismo hardware exacto entre celdas. Esto protege la
   comparabilidad, pero contradice la sugerencia amplia de usar varias PCs si no son homogéneas.

## Regla de decisión

Estado actual: **GO para piloto corto; NO-GO para corrida paper completa**.

El GO completo requiere, antes de observar nuevos resultados: decidir 5-fold, decidir binario
versus cuatro clases en BNCI2014_001, congelar la procedencia de recetas y fortalecer la identidad
de entorno/datos. Los demás faltantes pueden implementarse antes del análisis final sin repetir el
entrenamiento, siempre que los JSON con predicciones, historial y conteos se conserven.
