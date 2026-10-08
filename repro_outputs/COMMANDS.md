# Comandos ejecutados

Todos los comandos se ejecutaron desde la raíz del repositorio.

```bash
conda run -n eeg-diffusion pytest
conda run -n eeg-diffusion ruff check .

MNE_DONTWRITE_HOME=true PYTHONPATH=src \
  conda run -n eeg-diffusion python scripts/preflight.py

MNE_DONTWRITE_HOME=true PYTHONPATH=src \
  conda run -n eeg-diffusion python scripts/run_paper.py \
  --phase all --plan-only --device cpu \
  --output-dir /tmp/clei-scientific-audit.BVbjNz/plan

MNE_DONTWRITE_HOME=true PYTHONPATH=src \
  conda run -n eeg-diffusion python scripts/run_statistics.py \
  --results-dir /tmp/clei-scientific-audit.BVbjNz/plan

MNE_DONTWRITE_HOME=true PYTHONPATH=src \
  conda run -n eeg-diffusion python scripts/run_experiments.py \
  --dataset MI-OpenBCI --subjects S02 --models EEGNet \
  --protocols within_split --conditions full --seeds 0 \
  --epochs 1 --ica-policy none --device cpu \
  --output-dir /tmp/clei-scientific-audit.BVbjNz/smoke

MNE_DONTWRITE_HOME=true PYTHONPATH=src \
  conda run -n eeg-diffusion python scripts/run_statistics.py \
  --results-dir /tmp/clei-scientific-audit.BVbjNz/smoke \
  --allow-incomplete
```

La primera invocación directa de los scripts sin `PYTHONPATH=src` falló porque el entorno usado
para auditar no tenía instalado este repositorio. Esto no invalida `environment.yml`, que declara
`pip -e .`, pero debe verificarse durante el preflight de la PC objetivo.
