# Deep EEG benchmark para CLEI 2026

Pipeline reproducible para comparar exclusivamente EEGNet, FBCNet, ShallowConvNet,
EEGConformer y EEGInceptionMI. El codigo elimina CSP del benchmark nuevo, repite cada experimento
con cinco semillas y aplica Wilcoxon pareado con correccion de Holm usando al sujeto como unidad
estadistica.

## Datasets y alcance

- MI-OpenBCI: dataset low-cost principal, motor imagery frente a rest.
- Zhou2020: validacion research-grade task-matched, right hand frente a rest y siete sesiones.
- Tavakolan2017: validacion research-grade task-matched, right hand frente a rest, con cuatro
  sesiones y un costo computacional manejable.
- AlexMI: prueba rapida de instalacion, no resultado principal.

Los resultados entre datasets se reportan por separado. No deben interpretarse como una estimacion
causal del efecto de usar hardware low-cost, porque las tareas, sujetos y protocolos de adquisicion
no son identicos.

## Instalacion en la PC de Cayetano

Se recomienda Ubuntu o WSL2 con una GPU NVIDIA y drivers recientes. Desde una terminal:

```bash
git clone https://github.com/miguel-isidro05/clei-2026-deep-eeg-benchmark.git
cd clei-2026-deep-eeg-benchmark
conda env create -f environment.yml
conda activate deep-eeg-clei
```

En PowerShell, luego de copiar los archivos MAT de MI-OpenBCI a una carpeta local:

```powershell
$env:CLEI_DATA_DIR="D:\datos\Database-MIOpenBCI-main"
$env:MNE_DATA="D:\datos\mne"
python scripts\preflight.py --require-cuda
python scripts\download_moabb.py --datasets Zhou2020 Tavakolan2017
```

En Linux o WSL2:

```bash
export CLEI_DATA_DIR=/datos/Database-MIOpenBCI-main
export MNE_DATA=/datos/mne
python scripts/preflight.py --require-cuda
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

## Perfil completo del paper

El proceso es reanudable: una celda ya terminada se omite. Ejecute las fases dentro de `tmux` o
de un job persistente. No use `--overwrite` al reanudar.

```bash
python scripts/run_paper.py --phase all --plan-only
python scripts/run_paper.py --phase primary --device cuda
python scripts/run_paper.py --phase external --device cuda
python scripts/run_paper.py --phase ica-sensitivity --device cuda
python scripts/profile_latency.py --device cuda
python scripts/run_statistics.py
```

`primary` ejecuta MI-OpenBCI sin ICA, con trials completos, split estratificado 70/30, cinco folds
within-session, LOSO y el experimento separado de ventanas. `external` ejecuta por separado el
baseline 70/30 y leave-one-session-out en Zhou2020 y Tavakolan2017, tambien sin ICA.
`ica-sensitivity` repite el bloque principal 70/30 con la politica exploratoria de kurtosis.

El perfil completo contiene 3.350 celdas y ronda 12.150 entrenamientos, porque las celdas
within-session y externas contienen varios folds. Cada celda externa contiene un fold por
sesion para within-session y otro para cross-session. Antes de lanzarlo, mida una muestra pequena
con 3 a 5 epochs en otro directorio y estime el
tiempo de 300 epochs. Si hay varias GPU o PCs que comparten la carpeta de resultados, divida los
sujetos sin solaparlos:

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
- `results/statistics/augmentation_wilcoxon_holm.csv`: ventanas contra control center-crop.
- `results/statistics/completeness.csv`: semillas faltantes; la inferencia se bloquea si falta una.
- `results/statistics/expected_cell_audit.csv`: producto esperado del perfil; detecta incluso
  modelos, sujetos o condiciones totalmente ausentes.
- `results/statistics/sample_accounting.csv`: trials, ejemplos, folds y conteos de clase por
  celda, incluida la multiplicacion producida por ventanas.
- `results/latency/`: latencia controlada de forward pass.

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
