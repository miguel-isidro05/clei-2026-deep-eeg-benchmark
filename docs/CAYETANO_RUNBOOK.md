# Runbook operativo para la PC de Cayetano

## 1. Preparacion

1. Instalar Git, Miniconda, `wget` y el driver NVIDIA. Confirmar `nvidia-smi`.
2. Clonar el repositorio.
3. Ejecutar `CAYETANO=1 bash setup.sh`.
4. Activar `deep-eeg-clei`.
5. No continuar si el instalador no termina en `setup=OK` y `preflight=OK`.

El adjunto público 42 es una copia exacta del sujeto `004`; no es un sujeto `001` válido. El
instalador lo deja en cuarentena. Souza2023 se ejecuta únicamente con `002`–`006`. No renombre ni
duplique el archivo 004 y no cree un `001.edf`.

`Tavakolan2017` usa el lector BCI2000 que MOABB importa de forma opcional. El proyecto lo fija a
un commit exacto de su repositorio oficial y `preflight.py` comprueba su presencia antes de
descargar o entrenar. Si el entorno ya existia, actualicelo con
`python -m pip install -e '.[dev]'`.

## 2. Piloto temporal separado

Use otro directorio para que el piloto nunca se mezcle con el paper:

```bash
python scripts/run_experiments.py --dataset MI-OpenBCI --subjects S02 \
  --models EEGNet FBCNet ShallowConvNet EEGConformer EEGInceptionMI \
  --protocols within_split --conditions full --seeds 0 --epochs 3 \
  --ica-policy none --device cuda --output-dir results_pilot
```

Ejecute también el piloto de Souza2023, que verifica los cinco modelos en la tarea izquierda frente
a derecha:

```bash
python scripts/run_experiments.py --dataset Souza2023 --subjects 002 \
  --models EEGNet FBCNet ShallowConvNet EEGConformer EEGInceptionMI \
  --protocols within_split cross_session --conditions full --seeds 0 --epochs 3 \
  --ica-policy none --device cuda --output-dir results_souza_pilot
```

Revise `nvidia-smi`, el tiempo de cada epoch en `training_history` y el espacio en disco. Multiplique
el tiempo observado de forma conservadora antes de lanzar 300 epochs.

Valide por separado el lector BCI2000 y las cuatro sesiones externas:

```bash
python scripts/run_experiments.py --dataset Tavakolan2017 --subjects 1 \
  --models EEGNet --protocols within_split --conditions full --seeds 0 --epochs 1 \
  --ica-policy none --device cuda --output-dir results_tavakolan_pilot
```

## 3. Corrida paper

La forma recomendada en la RTX A6000 doble es el script completo, porque guarda consola y resumen:

```bash
bash run_cayetano.sh
```

El script escribe `results/logs/paper-*.log`, `results/EXPERIMENT_LOG.md`, tablas, latencia y
figuras, y actualiza `manuscript/revision_v2/generated_results.tex`. Si prefiere ejecutar
manualmente, use cada fase por separado dentro de `tmux`, `screen` o un job persistente:

```bash
python scripts/run_paper.py --phase all --plan-only
python scripts/run_paper.py --phase peterson --device cuda:0
python scripts/run_paper.py --phase souza --device cuda:1
python scripts/run_paper.py --phase external --device cuda
python scripts/run_paper.py --phase ica-sensitivity --device cuda
```

Regenerar primero el manifiesto `--phase all --plan-only` garantiza que el gate use el perfil
vigente. No edite ni recicle manualmente manifiestos de otra revisión.

Puede repetir exactamente el mismo comando tras una interrupcion. Se omiten celdas compatibles y
se recuperan folds finalizados. Si el codigo, epochs o receta cambiaron, el programa se detiene en
vez de mezclar resultados; use una nueva carpeta o `--overwrite` conscientemente.

## 4. Cierre y comprobaciones

```bash
python scripts/profile_latency.py --device cuda
python scripts/run_statistics.py
python scripts/generate_figures.py
python scripts/generate_manuscript_results.py
python scripts/write_run_report.py --status completed
pytest
ruff check .
```

No considere lista la corrida si `run_statistics.py` devuelve error, si existe un `present=False`
o un `expected=False` en `expected_cell_audit.csv`, o si existe un `complete=False` en
`completeness.csv`. El análisis confirmatorio también exige el manifiesto `--phase all` o los tres
manifiestos de fase. Conserve juntos
`results/cells`, `results/manifests`, `results/statistics`, `results/quality`, `results/latency`,
`results/figures`, `results/logs` y `results/EXPERIMENT_LOG.md`.

Si la sesión se interrumpe, vuelva a ejecutar `bash run_cayetano.sh`. Las celdas y folds
compatibles se omiten. No mezcle una carpeta `results` creada con otro commit o perfil: el
fingerprint y el auditor la rechazarán.
