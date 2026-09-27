# Runbook operativo para la PC de Cayetano

## 1. Preparacion

1. Instalar Git, Miniconda y el driver NVIDIA. Confirmar `nvidia-smi`.
2. Clonar el repositorio y crear el entorno con `conda env create -f environment.yml`.
3. Activar `deep-eeg-clei`.
4. Definir `CLEI_DATA_DIR` y `MNE_DATA` como se muestra en el README.
5. Ejecutar `python scripts/preflight.py --require-cuda`. No continuar si no termina en `preflight=OK`.
6. Descargar primero los datos con
   `python scripts/download_moabb.py --datasets Zhou2020 Tavakolan2017`.

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

Revise `nvidia-smi`, el tiempo de cada epoch en `training_history` y el espacio en disco. Multiplique
el tiempo observado de forma conservadora antes de lanzar 300 epochs.

## 3. Corrida paper

Ejecute cada fase por separado dentro de `tmux`, `screen` o un job persistente:

```bash
python scripts/run_paper.py --phase primary --device cuda
python scripts/run_paper.py --phase external --device cuda
python scripts/run_paper.py --phase ica-sensitivity --device cuda
```

Puede repetir exactamente el mismo comando tras una interrupcion. Se omiten celdas compatibles y
se recuperan folds finalizados. Si el codigo, epochs o receta cambiaron, el programa se detiene en
vez de mezclar resultados; use una nueva carpeta o `--overwrite` conscientemente.

## 4. Cierre y comprobaciones

```bash
python scripts/profile_latency.py --device cuda
python scripts/run_statistics.py
pytest
ruff check .
```

No considere lista la corrida si `run_statistics.py` devuelve error, si existe un `present=False`
en `expected_cell_audit.csv` o si existe un `complete=False` en `completeness.csv`. Conserve juntos
`results/cells`, `results/manifests`, `results/statistics` y `results/latency`.
