# Plan de implementación Peterson Diffusion V11

Fecha: 2026-10-08

Rama: `exp/peterson-diffusion-v11`

Base congelada: `feat/peterson-journal-v10@6c116d5`

Especificación: `docs/superpowers/specs/2026-10-08-peterson-diffusion-v11-design.md`

## 1. Resultado esperado

Implementar un banco experimental aislado para Peterson que permita ejecutar, auditar y decidir las
olas 0-5 del estudio MCDD sin modificar las recetas ni los artefactos V10. La primera entrega
ejecutable comprenderá la infraestructura común y las olas 0-1; las olas posteriores solo se
habilitarán cuando el archivo de decisión de la ola anterior indique `advance`.

La implementación termina cuando:

- los tests unitarios, de integración y de reproducibilidad pasan;
- cada shard produce un manifiesto exacto, celdas atómicas y reanudables;
- dos shards forman una partición disjunta y completa del plan;
- el auditor rechaza leakage, recetas obsoletas, celdas incompletas y hashes inesperados;
- el launcher de Cayetano usa `cuda:0` y `cuda:1`, propaga fallos y genera un export verificable;
- V10 conserva el commit, código y resultados sin cambios.

## 2. Decisiones de implementación

### 2.1 Aislamiento

El paquete vivirá bajo `experiments/peterson_diffusion_v11/`. Los scripts públicos estarán en
`scripts/` y el launcher en la raíz porque esos son los puntos de entrada ya usados en Cayetano. No
se modificarán `src/deepbench/evaluation.py`, `src/deepbench/models.py`, los manifiestos V10 ni
`run_peterson_finalize.sh`.

Los componentes estables de V10 se reutilizarán mediante sus interfaces existentes: carga del
dataset, preprocesamiento entrenado solo con train, augmentación posterior al split, agregación por
trial y métricas. El runner V11 añadirá su propio fingerprint, manifiesto y formato de celda.

### 2.2 TDD y commits pequeños

Cada tarea comienza con un test que falla, continúa con la implementación mínima y termina con un
commit independiente. No se implementará una ola cuya puerta de avance dependa de resultados aún no
observados; sí se dejarán definidos y probados sus módulos comunes.

### 2.3 Reproducibilidad

La identidad de una celda incluirá `wave`, arquitectura, objetivo, condición, sujeto, seed y, cuando
corresponda, severidad y seed de corrupción. El fingerprint cubrirá el paquete experimental, los
scripts V11, la configuración, el commit base V10, el hash del dataset y el hash de los splits.

Los archivos JSON y checkpoints se escribirán primero a un temporal en el mismo directorio y se
renombrarán atómicamente. Una celda solo se omite al reanudar si pasa validación de esquema,
fingerprint, receta y estado `completed`.

## 3. Contratos y conteos congelados

Sujetos Peterson: `S02-S10` y `S12` (10).

Condiciones: `full`, `center_x6`, `overlap`.

| Ola | Plan antes de entrenar | Celdas esperadas |
|---|---|---:|
| 0 | 5 modelos × 5 seeds × 10 sujetos × `center_x6` | 250 |
| 1 | 4 backbones × 2 formulaciones × 3 condiciones × 1 seed × 10 sujetos | 240 |
| 2 | 3 objetivos × 3 condiciones × 3 seeds × 10 sujetos | 270 |
| 3 | ganador + control × condiciones que avancen × 5 seeds × 10 sujetos | parametrizado por decisión |
| 4 | checkpoints de ola 3 × 4 familias de corrupción × severidades predeclaradas | parametrizado por manifiesto |
| 5 | checkpoints congelados × K `{1,4,8,16}` + calibración y coste | parametrizado por manifiesto |

Las olas 3-5 no usarán un número codificado manualmente: el manifiesto se derivará del archivo de
decisión firmado de la ola previa y guardará el conteo exacto antes de ejecutar.

## 4. Tareas de implementación

### Tarea 1. Crear el paquete experimental y sus contratos

Archivos:

- crear `experiments/peterson_diffusion_v11/__init__.py`;
- crear `experiments/peterson_diffusion_v11/config.py`;
- crear `experiments/peterson_diffusion_v11/identity.py`;
- crear `experiments/peterson_diffusion_v11/io.py`;
- crear `tests/diffusion_v11/test_config.py`;
- crear `tests/diffusion_v11/test_identity.py`;
- actualizar `.gitignore` únicamente para `results_diffusion_v11/` si no existe una regla
  equivalente.

Pruebas que deben fallar primero:

1. rechazar una configuración con LOSO, dataset distinto de Peterson o sujeto fuera del conjunto;
2. producir la misma identidad para diccionarios semánticamente iguales;
3. cambiar el fingerprint cuando cambia un archivo experimental o una configuración;
4. preservar los hashes declarados de `6c116d5` y de las recetas V10 referenciadas;
5. no aceptar como completa una celda parcial o con fingerprint anterior.

Implementación mínima:

- dataclasses/enums para olas, condiciones, formulaciones y objetivos;
- serialización canónica JSON;
- hashing SHA-256 por streaming;
- escritura atómica de JSON y checkpoints;
- captura de Git, estado limpio/sucio y versiones del entorno.

Verificación:

```bash
conda run -n eeg-diffusion pytest -q \
  tests/diffusion_v11/test_config.py \
  tests/diffusion_v11/test_identity.py
```

Commit: `feat(v11): establish experiment identity and integrity contracts`

### Tarea 2. Congelar manifiestos y splits por trial

Archivos:

- crear `experiments/peterson_diffusion_v11/manifests.py`;
- crear `experiments/peterson_diffusion_v11/splits.py`;
- crear `experiments/peterson_diffusion_v11/configs/wave_00.json`;
- crear `experiments/peterson_diffusion_v11/configs/wave_01.json`;
- crear `tests/diffusion_v11/test_manifests.py`;
- crear `tests/diffusion_v11/test_splits.py`.

Pruebas que deben fallar primero:

1. ola 0 genera exactamente 250 identidades únicas;
2. ola 1 genera exactamente 240 identidades únicas;
3. shards 0 y 1 son disjuntos y su unión equivale al manifiesto completo;
4. todas las ventanas de un trial quedan en el mismo fold;
5. el hash de split es independiente del orden de entrada;
6. los splits V11 coinciden con los splits V10 para igual sujeto y seed;
7. el auditor de plan rechaza duplicados y configuraciones no declaradas.

Implementación mínima:

- generar manifiestos ordenados de forma estable;
- encapsular la receta V10 de split sin importar funciones privadas;
- registrar índices de trial antes de crear ventanas;
- escribir `manifests/expected.json` antes de cualquier entrenamiento.

Verificación:

```bash
conda run -n eeg-diffusion pytest -q \
  tests/diffusion_v11/test_manifests.py \
  tests/diffusion_v11/test_splits.py
```

Commit: `feat(v11): freeze wave manifests and trial-level splits`

### Tarea 3. Implementar la ola 0 como control comparable

Archivos:

- crear `experiments/peterson_diffusion_v11/runners/baseline.py`;
- crear `experiments/peterson_diffusion_v11/runners/__init__.py`;
- crear `scripts/run_diffusion_v11.py`;
- crear `tests/diffusion_v11/test_wave0_runner.py`.

Pruebas que deben fallar primero:

1. `--plan-only --wave 0` reporta 250 celdas;
2. `center_x6` crea seis copias de la misma ventana central, sin multiplicar trials de test;
3. `center_x6` y `overlap` tienen iguales ejemplos de train, batches y actualizaciones;
4. predicciones de ventanas se agregan una sola vez por trial;
5. un smoke sintético escribe una celda completa y la segunda ejecución la omite;
6. una celda truncada se vuelve a ejecutar;
7. CSP conserva una marca explícita de determinismo cuando las seeds producen la misma receta.

Implementación mínima:

- reutilizar los modelos y la receta de entrenamiento V10;
- limitar esta ola a `within_session/center_x6/ica-none`;
- guardar conteos de trials, ventanas, batches y actualizaciones;
- escribir predicciones por trial y métricas completas.

Verificación:

```bash
conda run -n eeg-diffusion pytest -q tests/diffusion_v11/test_wave0_runner.py
conda run -n eeg-diffusion python scripts/run_diffusion_v11.py \
  --wave 0 --plan-only --output-dir /tmp/peterson-v11-plan
```

Commit: `feat(v11): add compute-matched Peterson window control`

### Tarea 4. Implementar bloques comunes y pares comparables

Archivos:

- crear `experiments/peterson_diffusion_v11/models/common.py`;
- crear `experiments/peterson_diffusion_v11/models/residual.py`;
- crear `experiments/peterson_diffusion_v11/models/filterbank.py`;
- crear `experiments/peterson_diffusion_v11/models/smooth_basis.py`;
- crear `experiments/peterson_diffusion_v11/models/conformer.py`;
- crear `experiments/peterson_diffusion_v11/models/factory.py`;
- crear `experiments/peterson_diffusion_v11/models/__init__.py`;
- crear `tests/diffusion_v11/test_models.py`;
- crear `tests/diffusion_v11/test_parameter_matching.py`.

Pruebas que deben fallar primero:

1. cada denoiser transforma `(B,C,T)` en `(B,C,T)` para `T=256` y `T=512`;
2. ninguna arquitectura reduce la resolución temporal de salida;
3. timestep, clase y máscara modifican la salida;
4. forward/backward produce valores y gradientes finitos;
5. la contraparte discriminativa comparte el backbone correspondiente;
6. la diferencia de parámetros de cada par queda bajo 10%, excluyendo solo la cabeza declarada;
7. seed idéntica produce inicialización y salida idénticas;
8. SmoothBasis mantiene la base fija y aprende únicamente coeficientes.

Implementación mínima:

- embeddings sinusoidales de timestep;
- embeddings de clase y resumen de máscara;
- modulación FiLM común;
- cuatro backbones compactos con capacidad parametrizada;
- cabezas generativa y discriminativa separadas;
- contador de parámetros por componentes.

Verificación:

```bash
conda run -n eeg-diffusion pytest -q \
  tests/diffusion_v11/test_models.py \
  tests/diffusion_v11/test_parameter_matching.py
```

Commit: `feat(v11): implement parameter-matched MCDD backbones`

### Tarea 5. Implementar difusión, energías y objetivos

Archivos:

- crear `experiments/peterson_diffusion_v11/diffusion.py`;
- crear `experiments/peterson_diffusion_v11/energy.py`;
- crear `experiments/peterson_diffusion_v11/objectives.py`;
- crear `tests/diffusion_v11/test_diffusion.py`;
- crear `tests/diffusion_v11/test_energy.py`;
- crear `tests/diffusion_v11/test_objectives.py`.

Pruebas que deben fallar primero:

1. `q_sample` coincide con la ecuación para valores controlados;
2. ambas clases candidatas usan exactamente el mismo timestep y ruido;
3. la energía ignora elementos marcados como no observables;
4. `K={1,4,8,16}` produce formas y promedios correctos;
5. `noise_only`, `noise_rank` y `noise_rank_consistency` activan solo sus términos declarados;
6. ranking es cero cuando se satisface el margen;
7. consistencia es simétrica y finita;
8. el cálculo en minibatches no altera el resultado dentro de tolerancia.

Implementación mínima:

- schedule cosine o lineal fijado en configuración, sin búsqueda en test;
- generador de ruido explícito por seed;
- common random numbers entre clases;
- energía media sobre elementos observados;
- ranking hinge y JS de consistencia;
- softmax de energías con temperatura positiva.

Verificación:

```bash
conda run -n eeg-diffusion pytest -q \
  tests/diffusion_v11/test_diffusion.py \
  tests/diffusion_v11/test_energy.py \
  tests/diffusion_v11/test_objectives.py
```

Commit: `feat(v11): add conditional diffusion energy objectives`

### Tarea 6. Implementar entrenamiento y evaluación within-session

Archivos:

- crear `experiments/peterson_diffusion_v11/training.py`;
- crear `experiments/peterson_diffusion_v11/evaluation.py`;
- crear `experiments/peterson_diffusion_v11/runners/wave_runner.py`;
- crear `tests/diffusion_v11/test_training.py`;
- crear `tests/diffusion_v11/test_evaluation.py`;
- crear `tests/diffusion_v11/test_wave1_runner.py`.

Pruebas que deben fallar primero:

1. normalización, ICA si alguna vez se habilita y calibración se ajustan solo con train/validation;
2. no existe acceso a etiquetas de test durante entrenamiento o selección;
3. augmentación ocurre después del split por trial;
4. la evaluación devuelve una predicción por trial;
5. early stopping restaura el mejor estado de validación;
6. checkpoints contienen pesos, configuración, hashes, curvas y estado RNG;
7. un checkpoint cargado reproduce logits/energías de prueba;
8. ola 1 planifica 240 celdas y ejecuta un smoke sintético por cada backbone/formulación.

Implementación mínima:

- loop PyTorch explícito con AMP configurable, clipping y control de no finitos;
- validación interna fija dentro de train;
- presupuesto idéntico de epochs/updates para pares comparables;
- corrupciones de training compartidas entre MCDD y control discriminativo;
- checkpoint por sujeto/seed/configuración;
- métricas por trial: accuracy, kappa, balanced accuracy, macro-F1, AUROC, NLL y Brier.

Verificación:

```bash
conda run -n eeg-diffusion pytest -q \
  tests/diffusion_v11/test_training.py \
  tests/diffusion_v11/test_evaluation.py \
  tests/diffusion_v11/test_wave1_runner.py
```

Commit: `feat(v11): train and evaluate diffusion classifiers within session`

### Tarea 7. Añadir launcher de dos GPUs y reanudación segura

Archivos:

- crear `run_diffusion_cayetano.sh`;
- crear `experiments/peterson_diffusion_v11/run_report.py`;
- crear `tests/diffusion_v11/test_launcher.py`;
- actualizar `docs/CAYETANO_RUNBOOK.md` con una sección V11 separada.

Pruebas que deben fallar primero:

1. `bash -n run_diffusion_cayetano.sh` pasa;
2. el launcher rechaza menos de dos dispositivos cuando se solicita modo dual;
3. crea dos procesos con shards 0/2 y 1/2;
4. espera ambos PIDs y devuelve fallo si cualquiera falla;
5. no genera `run_complete.json` ante un shard incompleto;
6. reanuda celdas válidas y vuelve a ejecutar las inválidas;
7. el reporte incluye commit, dispositivos, manifiesto y tiempo.

Implementación mínima:

- variables `DIFFUSION_WAVE`, `DIFFUSION_RESULTS_DIR` y
  `DIFFUSION_DEVICES='cuda:0 cuda:1'`;
- preflight de dos GPUs y dataset Peterson;
- logs sin buffering por shard;
- trap que siempre escribe estado final;
- auditoría antes de estadísticas o export.

Verificación:

```bash
bash -n run_diffusion_cayetano.sh
conda run -n eeg-diffusion pytest -q tests/diffusion_v11/test_launcher.py
```

Commit: `feat(v11): orchestrate resumable dual-GPU waves`

### Tarea 8. Construir auditor científico y puertas de decisión

Archivos:

- crear `experiments/peterson_diffusion_v11/audit.py`;
- crear `experiments/peterson_diffusion_v11/statistics.py`;
- crear `experiments/peterson_diffusion_v11/decision.py`;
- crear `scripts/check_diffusion_v11.py`;
- crear `scripts/analyze_diffusion_v11.py`;
- crear `tests/diffusion_v11/test_audit.py`;
- crear `tests/diffusion_v11/test_statistics.py`;
- crear `tests/diffusion_v11/test_decision.py`.

Pruebas que deben fallar primero:

1. detectar celda faltante, extra, duplicada, truncada u obsoleta;
2. detectar mezcla de trials entre folds y conteos de ventanas incompatibles;
3. rechazar NaN, colapso de clase y checkpoint no reproducible;
4. promediar seeds dentro de sujeto antes de inferencia;
5. Wilcoxon, intervalo, rank-biserial y Holm coinciden con ejemplos conocidos;
6. H1 solo avanza con `>=0.02` y al menos 7/10 direcciones favorables;
7. H2-H4 producen `advance`, `revise` o `stop` sin cambiar umbrales;
8. una comparación no predeclarada queda marcada `exploratory`;
9. el archivo de decisión incluye hashes de entradas y no es mutable silenciosamente.

Implementación mínima:

- esquema estricto de celda/manifiesto;
- comparaciones pareadas por participante;
- bootstrap o permutación determinista para intervalos, según el contrato congelado;
- corrección Holm por familia;
- `decision.json` legible por la siguiente ola.

Verificación:

```bash
conda run -n eeg-diffusion pytest -q \
  tests/diffusion_v11/test_audit.py \
  tests/diffusion_v11/test_statistics.py \
  tests/diffusion_v11/test_decision.py
```

Commit: `feat(v11): audit evidence and enforce wave gates`

### Tarea 9. Preparar ola 4: degradaciones compartidas y control Riemanniano

Esta tarea se implementa después de que ola 3 produzca una configuración congelada.

Archivos:

- crear `experiments/peterson_diffusion_v11/corruptions.py`;
- crear `experiments/peterson_diffusion_v11/missing_baselines.py`;
- crear `experiments/peterson_diffusion_v11/configs/wave_04.json` a partir de `decision.json`;
- crear `tests/diffusion_v11/test_corruptions.py`;
- crear `tests/diffusion_v11/test_missing_baselines.py`;
- crear `tests/diffusion_v11/test_wave4_runner.py`.

Pruebas que deben fallar primero:

1. misma trial/seed/severidad produce la misma corrupción para todos los modelos;
2. distinta seed de corrupción produce una realización distinta;
3. cada severidad altera exactamente canales, duración o SNR declarados;
4. test degradado no actualiza pesos, normalización ni temperatura;
5. accuracy-severity AUC integra puntos en orden y escala común;
6. el control EM-SCM+MDRM produce covarianzas SPD y una predicción finita;
7. el caso sin valores ausentes coincide con el pipeline Riemanniano ordinario dentro de tolerancia.

Implementación mínima:

- bancos de corrupciones precomputados y hasheados;
- canales ausentes, dropout temporal, ruido, clipping e impulsos;
- EM-SCM y MDRM mediante `pyriemann 0.11`, declarado como dependencia directa y fijado en el
  entorno;
- evaluación de checkpoints sin reentrenamiento.

Verificación:

```bash
conda run -n eeg-diffusion pytest -q \
  tests/diffusion_v11/test_corruptions.py \
  tests/diffusion_v11/test_missing_baselines.py \
  tests/diffusion_v11/test_wave4_runner.py
```

Commit: `feat(v11): evaluate controlled low-cost EEG degradations`

### Tarea 10. Preparar ola 5: calibración, rechazo y coste

Esta tarea se implementa después de auditar ola 4.

Archivos:

- crear `experiments/peterson_diffusion_v11/calibration.py`;
- crear `experiments/peterson_diffusion_v11/cost.py`;
- crear `experiments/peterson_diffusion_v11/configs/wave_05.json`;
- crear `tests/diffusion_v11/test_calibration.py`;
- crear `tests/diffusion_v11/test_cost.py`;
- crear `tests/diffusion_v11/test_wave5_runner.py`.

Pruebas que deben fallar primero:

1. temperatura se ajusta exclusivamente con validation;
2. calibración no modifica la clase predicha salvo empate numérico;
3. risk-coverage está ordenada por confianza y AURC coincide con un ejemplo manual;
4. `K={1,4,8,16}` comparte bancos de ruido anidados para comparación justa;
5. contador de forwards por predicción coincide con `2K` para dos clases;
6. latencia separa preprocess, forward/energía y agregación;
7. memoria máxima, parámetros y throughput tienen unidades explícitas.

Implementación mínima:

- temperature scaling escalar;
- ECE, Brier, NLL, risk-coverage y AURC;
- profiler con warm-up, sincronización CUDA y repeticiones;
- reporte de precisión/coste para seleccionar K.

Verificación:

```bash
conda run -n eeg-diffusion pytest -q \
  tests/diffusion_v11/test_calibration.py \
  tests/diffusion_v11/test_cost.py \
  tests/diffusion_v11/test_wave5_runner.py
```

Commit: `feat(v11): quantify calibration rejection and inference cost`

### Tarea 11. Figuras, readout y export verificable

Archivos:

- crear `experiments/peterson_diffusion_v11/figures.py`;
- crear `experiments/peterson_diffusion_v11/export.py`;
- crear `scripts/finalize_diffusion_v11.py`;
- crear `tests/diffusion_v11/test_figures.py`;
- crear `tests/diffusion_v11/test_export.py`.

Pruebas que deben fallar primero:

1. no generar figuras si el auditor no marca la ola completa;
2. toda figura tiene un CSV/JSON fuente y hash;
3. ejes, unidades, n y definición de barras de error quedan en metadatos;
4. `run_complete.json` enlaza manifiesto, decisión, estadísticas y figuras;
5. export incluye solo código de identidad, resultados de la ola, configuración y logs necesarios;
6. SHA-256 valida después de extraer el archivo en un temporal.

Implementación mínima:

- accuracy/kappa por sujeto, distribución por seeds, Pareto accuracy-cost, curvas de severidad,
  risk-coverage y sensibilidad a K;
- formatos PNG 300 dpi, PDF vectorial y datos tabulares;
- readout en Markdown que separa hechos, inferencias y claims no permitidos;
- tarball y checksum portable cuyo archivo `.sha256` use solo el basename.

Verificación:

```bash
conda run -n eeg-diffusion pytest -q \
  tests/diffusion_v11/test_figures.py \
  tests/diffusion_v11/test_export.py
```

Commit: `feat(v11): finalize traceable figures and wave exports`

### Tarea 12. Verificación integral antes de subir la rama

Archivos revisados: todos los anteriores; no se añaden resultados grandes al repositorio.

Comandos:

```bash
conda run -n eeg-diffusion pytest -q
conda run -n eeg-diffusion ruff check \
  experiments/peterson_diffusion_v11 scripts tests/diffusion_v11
bash -n run_diffusion_cayetano.sh
conda run -n eeg-diffusion python scripts/run_diffusion_v11.py \
  --wave 0 --plan-only --output-dir /tmp/peterson-v11-wave0
conda run -n eeg-diffusion python scripts/run_diffusion_v11.py \
  --wave 1 --plan-only --output-dir /tmp/peterson-v11-wave1
git diff --check
git status --short
```

Además:

- revisar que `git diff feat/peterson-journal-v10 -- src/deepbench run_peterson_finalize.sh` esté vacío
  salvo cambios expresamente aprobados;
- ejecutar smoke CPU con datos sintéticos para las ocho variantes de ola 1;
- ejecutar smoke GPU para una celda por shard en Cayetano antes de lanzar la ola completa;
- revisar el diff de seguridad y correr `pip-audit` si está disponible, sin instalar paquetes no
  aprobados durante la auditoría;
- hacer una revisión final de código, reproducibilidad y leakage.

Commit final de correcciones, si se necesita: `fix(v11): close verification findings`

## 5. Secuencia operativa por resultados

La primera ejecución remota será ola 0. Solo después de transferir y auditar su export se decidirá
si ola 1 usa `overlap` como condición primaria o `full` como primaria. Aunque el código de ola 1
admita las tres condiciones para la comparación exploratoria, la puerta H1 define la interpretación
y evita reescribir la hipótesis con los resultados a la vista.

Después de cada ola:

1. `check_diffusion_v11.py` valida integridad y completitud;
2. `analyze_diffusion_v11.py` genera estadística y `decision.json`;
3. `finalize_diffusion_v11.py` genera figuras, readout, export y SHA-256;
4. el usuario transfiere ambos archivos al Mac;
5. se revisa la evidencia y se versiona únicamente la configuración de la siguiente ola.

## 6. Comandos remotos que se entregarán tras la implementación

La forma final será:

```bash
cd /home/imiguel/Desktop/clei-2026-deep-eeg-benchmark
git fetch origin
git switch --track origin/exp/peterson-diffusion-v11
git pull --ff-only
conda activate eeg-diffusion

tmux new -s peterson_diffusion_v11
export CLEI_DATA_DIR='/home/imiguel/Desktop/clei-2026-deep-eeg-benchmark/data/mi-openbci'
export DIFFUSION_WAVE='0'
export DIFFUSION_DEVICES='cuda:0 cuda:1'
export DIFFUSION_RESULTS_DIR="results_diffusion_v11/wave_00_$(git rev-parse --short HEAD)"
bash run_diffusion_cayetano.sh
```

Estos comandos son una plantilla del contrato aprobado. El comando definitivo solo se entregará
después de que el commit remoto exista, los tests pasen y el SHA exacto quede verificado.

