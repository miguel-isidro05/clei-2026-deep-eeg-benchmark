# Deep EEG benchmark para CLEI 2026

Pipeline reproducible para comparar exclusivamente EEGNet, FBCNet, ShallowConvNet,
EEGConformer y EEGInceptionMI. El codigo elimina CSP del benchmark nuevo, repite cada experimento
con cinco semillas y aplica Wilcoxon pareado con correccion de Holm usando al sujeto como unidad
estadistica.

## Datasets y alcance

- MI-OpenBCI: dataset low-cost principal, motor imagery frente a rest.
- Souza2023: segundo dataset low-cost, mano izquierda frente a mano derecha. Se usan los cinco
  sujetos válidos `002`–`006`, con cuatro corridas por sujeto.
- Zhou2020: validacion research-grade task-matched, right hand frente a rest y siete sesiones.
- Tavakolan2017: validacion research-grade task-matched, right hand frente a rest, con cuatro
  sesiones y un costo computacional manejable.
- AlexMI: prueba rapida de instalacion, no resultado principal.

Los resultados entre datasets se reportan por separado. No deben interpretarse como una estimacion
causal del efecto de usar hardware low-cost, porque las tareas, sujetos y protocolos de adquisicion
no son identicos.

## Instalacion en la PC de Cayetano

Se recomienda Ubuntu o WSL2 con dos GPU NVIDIA y drivers recientes. El instalador crea el entorno,
descarga los datasets y ejecuta el preflight:

```bash
git clone https://github.com/miguel-isidro05/clei-2026-deep-eeg-benchmark.git
cd clei-2026-deep-eeg-benchmark
CAYETANO=1 bash setup.sh
conda activate deep-eeg-clei
bash run_cayetano.sh
```

El sitio de Souza2023 publica el mismo EDF en los adjuntos 42 y 45; ambos son el sujeto `004` y
tienen la misma huella SHA-256. `setup.sh` conserva el adjunto 42 en cuarentena y ejecuta Souza2023
únicamente con `002`–`006`. No se usa, sustituye ni busca un archivo `001.edf`. Los seis enlaces y
sus huellas están en `configs/souza2023_downloads.tsv`.

Si los datos ya están instalados en rutas externas, puede definirlos manualmente:

```bash
export CLEI_DATA_DIR=/datos/Database-MIOpenBCI-main
export SOUZA_DATA_DIR=/datos/souza2023
export MNE_DATA=/datos/mne
python scripts/preflight.py --require-cuda --min-cuda-devices 2
python scripts/download_moabb.py --datasets Zhou2020 Tavakolan2017
```

No suba los datasets al repositorio. MOABB conserva sus descargas en `MNE_DATA` y las reutiliza.

## Validacion corta antes del computo completo

Esta prueba descarga y entrena solo un sujeto de AlexMI durante una epoca:

```bash
python scripts/run_experiments.py \
  --dataset AlexMI \
  --subjects 1 \
  --models EEGNet \
  --protocols within_split \
  --conditions full \
  --seeds 0 \
  --epochs 1 \
  --ica-policy none \
  --device cuda
```

Después ejecute las pruebas del repositorio:

```bash
pytest
ruff check .
```

Antes de la matriz completa, valide también el lector y las cuatro sesiones de Tavakolan con una
sola celda de una época:

```bash
python scripts/run_experiments.py \
  --dataset Tavakolan2017 --subjects 1 --models EEGNet \
  --protocols within_split --conditions full --seeds 0 --epochs 1 \
  --ica-policy none --device cuda --output-dir results_tavakolan_pilot
```

## Perfil completo del paper

El proceso es reanudable: una celda ya terminada se omite. En la PC de Cayetano, la ruta
recomendada usa las dos GPU, conserva la consola completa y genera tablas, figuras y el bloque
LaTeX de resultados:

```bash
bash run_cayetano.sh
```

No use `--overwrite` al reanudar. La ejecución manual equivalente es:

```bash
python scripts/run_paper.py --phase all --plan-only
python scripts/run_paper.py --phase peterson --device cuda:0
python scripts/run_paper.py --phase souza --device cuda:1
python scripts/run_paper.py --phase external --device cuda
python scripts/run_paper.py --phase ica-sensitivity --device cuda
python scripts/profile_latency.py --device cuda
python scripts/run_statistics.py
python scripts/generate_figures.py
python scripts/generate_manuscript_results.py
python scripts/write_run_report.py
```

`run_statistics.py` solo genera inferencia confirmatoria cuando encuentra el manifiesto completo
del perfil y el conjunto de celdas presentes coincide exactamente con el esperado. Sin manifiesto,
con fases faltantes, celdas faltantes o celdas extra, se detiene. `--allow-incomplete` produce solo
salidas descriptivas marcadas como exploratorias y nunca tablas Wilcoxon-Holm.

`primary` reúne los bloques separados `peterson` y `souza`. Ambos ejecutan trials completos, split
estratificado 70/30, cinco folds within-session, LOSO y el experimento de ventanas. Souza2023 añade
leave-one-run-out como `cross_session`. `external` ejecuta por separado el
baseline 70/30 y leave-one-session-out en Zhou2020 y Tavakolan2017, tambien sin ICA.
`ica-sensitivity` repite el bloque 70/30 de los dos datasets low-cost con la politica exploratoria
de kurtosis.

El perfil completo contiene 4.725 celdas. El numero de entrenamientos es mayor que el numero de
celdas porque `within_session` y `cross_session` contienen varios folds. Cada celda externa contiene un baseline
`within_split` y un fold por sesion retenida para `cross_session`; `within_session` de cinco folds
se ejecuta en ambos datasets low-cost. Antes de lanzarlo, mida una muestra pequena con 3 a 5 epochs en otro
directorio y estime el
tiempo de 300 epochs. Si hay varias GPU o PCs que comparten la carpeta de resultados, divida los
sujetos sin solaparlos solo cuando todas usan la misma version del codigo, entorno, tipo de
dispositivo y modelo de GPU. No combine hardware heterogeneo en una misma corrida confirmatoria:
la validacion estadistica lo rechazara.

```bash
# Proceso/GPU 0
python scripts/run_paper.py --phase external --num-shards 2 --shard-index 0 --device cuda
# Proceso/GPU 1
python scripts/run_paper.py --phase external --num-shards 2 --shard-index 1 --device cuda
```

Cada fold terminado se guarda en `results/fold_cache`, por lo que una interrupcion dentro de una
celda multisesion no obliga a repetir los folds ya finalizados. Una celda existente solo se omite si
coinciden codigo, receta, epochs, dispositivo y configuracion; de otro modo la ejecucion se detiene
para evitar mezclar resultados incompatibles.

Para vigilar el avance:

```bash
find results/cells -name '*.json' | wc -l
nvidia-smi
```

Los pesos no se guardan por defecto, porque no son necesarios para reproducir las tablas y ocupan
mucho espacio. Agregue `--save-weights` solo si necesita archivar cada modelo.

## Salidas

- `results/cells/`: predicciones, metricas, seeds, hashes de indices e informes ICA por celda.
- `results/manifests/`: versiones, dispositivo y configuracion.
- `results/statistics/descriptive_subject_seed_variability.csv`: accuracy y kappa con dispersion
  entre sujetos, IC 95% y variabilidad de optimizacion separada.
- `results/statistics/paired_wilcoxon_holm.csv`: diez pares de modelos por familia.
- `results/statistics/augmentation_wilcoxon_holm.csv`: ventanas contra controles center-crop
  emparejados por computo (`nonoverlap-center_x2` y `overlap-center_x6`).
- `results/statistics/completeness.csv`: semillas faltantes; la inferencia se bloquea si falta una.
- `results/statistics/expected_cell_audit.csv`: producto esperado del perfil; detecta incluso
  modelos, sujetos o condiciones totalmente ausentes.
- `results/statistics/sample_accounting.csv`: trials, ejemplos, folds y conteos de clase por
  celda, incluida la multiplicacion producida por ventanas y las actualizaciones del optimizador.
- `results/statistics/statistics_manifest.json`: hashes que vinculan celdas, código estadístico y
  cada CSV.
- `results/figures/figures_manifest.json`: hashes de todos los PDF y del código de figuras.
- `results/latency/latency_manifest.json`: hash del CSV, perfil de medición, entorno y código de
  latencia.
- `results/latency/`: latencia controlada de forward pass.
- `results/quality/`: auditoria no destructiva de calidad de señal, conteos y flags.
- `results/figures/`: figuras PNG/PDF generadas solo tras pasar el gate confirmatorio.
- `results/logs/` y `results/EXPERIMENT_LOG.md`: consola completa y resumen auditable de corrida.
- `manuscript/revision_v2/generated_results.tex`: tablas y referencias a figuras, generado solo
  después del gate confirmatorio.

## Que significa la latencia

El perfil de latencia mide exclusivamente el forward pass del modelo ya cargado, despues de
warm-up, con el mismo dispositivo, numero de canales, longitud de entrada y batch para todos los
modelos. Informa mediana y percentil 95. No es latencia online total: excluye adquisicion, espera
de cuatro segundos para completar la ventana, transmision, filtrado, ajuste de ICA, interfaz y
actuacion. En el paper debe llamarse `model inference latency`, no `BCI response time`.

La especificacion metodologica completa esta en `docs/EXPERIMENT_SPEC.md` y el estado honesto de
cada observacion de los revisores en `docs/FEEDBACK_COVERAGE.md`.
La seleccion reproducible de datasets esta en `docs/MOABB_DATASET_INVENTORY.md` y la procedencia de
las recetas en `docs/HYPERPARAMETER_PROVENANCE.md`.
