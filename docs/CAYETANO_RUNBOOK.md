# Runbook operativo para Cayetano

## Objetivo

Ejecutar el perfil Peterson-only V10. No corre Souza2023, transferencia ni datasets MOABB externos.
El resultado esperado son 3,250 celdas y tablas estadisticas confirmatorias.

## Preparacion

```bash
cd /home/imiguel/Desktop/clei-2026-deep-eeg-benchmark
git fetch origin
git switch feat/peterson-journal-v10
git pull --ff-only

CAYETANO=1 bash setup.sh
conda activate deep-eeg-clei
```

Si `setup.sh` termina bien debe mostrar `setup=OK` y `preflight=OK`.

## Lanzar en tmux

```bash
tmux new -s clei_peterson_v10

cd /home/imiguel/Desktop/clei-2026-deep-eeg-benchmark
conda activate deep-eeg-clei

export CLEI_DATA_DIR='/home/imiguel/Desktop/clei-2026-deep-eeg-benchmark/data/mi-openbci'
export PETERSON_RESULTS_DIR="results_peterson_journal_v10_$(git rev-parse --short HEAD)"

bash run_cayetano.sh
```

Para salir sin cortar la corrida: `Ctrl-b`, luego `d`.

## Vigilar avance

```bash
cd /home/imiguel/Desktop/clei-2026-deep-eeg-benchmark
export PETERSON_RESULTS_DIR="results_peterson_journal_v10_$(git rev-parse --short HEAD)"

pgrep -af 'run_cayetano.sh|run_peterson_journal.py'
find "$PETERSON_RESULTS_DIR/cells" -name '*.json' | wc -l
nvidia-smi
tail -n 40 "$(ls -1t "$PETERSON_RESULTS_DIR"/logs/peterson-*.log | head -n 1)"
```

La corrida completa debe llegar a `3250/3250`. Si se corta, repita `bash run_cayetano.sh` con la
misma `PETERSON_RESULTS_DIR`; las celdas compatibles se saltan.

## Comprobar cierre

```bash
cd /home/imiguel/Desktop/clei-2026-deep-eeg-benchmark
conda activate deep-eeg-clei
export PETERSON_RESULTS_DIR="results_peterson_journal_v10_$(git rev-parse --short HEAD)"

pgrep -af 'run_cayetano.sh|run_peterson_journal.py' || echo "procesos=ninguno"
python scripts/check_peterson_journal.py --output-dir "$PETERSON_RESULTS_DIR"
test -f "$PETERSON_RESULTS_DIR/statistics/paired_wilcoxon_holm.csv" && echo "statistics=OK"
test -f "$PETERSON_RESULTS_DIR/latency/latency.json" && echo "latency=OK"
test -f "$PETERSON_RESULTS_DIR/quality/erd_ers.json" && echo "erd_ers=OK"
tail -n 40 "$(ls -1t "$PETERSON_RESULTS_DIR"/logs/peterson-*.log | head -n 1)"
```

## Exportar resultados

```bash
cd /home/imiguel/Desktop/clei-2026-deep-eeg-benchmark
export PETERSON_RESULTS_DIR="results_peterson_journal_v10_$(git rev-parse --short HEAD)"
export EXPORT_FILE="/home/imiguel/Desktop/clei-exports/clei-peterson-publication-v10-$(git rev-parse --short HEAD)-$(date -u +%Y%m%d).tar.gz"

mkdir -p /home/imiguel/Desktop/clei-exports
python scripts/export_peterson_publication.py \
  --results-dir "$PETERSON_RESULTS_DIR" \
  --output "$EXPORT_FILE"

ls -lh "$EXPORT_FILE" "${EXPORT_FILE}.sha256"
```

En la Mac, desde la carpeta donde quieras recibirlo:

```bash
scp hinton_2_cayetano:/home/imiguel/Desktop/clei-exports/clei-peterson-publication-v10-*.tar.gz .
scp hinton_2_cayetano:/home/imiguel/Desktop/clei-exports/clei-peterson-publication-v10-*.tar.gz.sha256 .
shasum -a 256 -c clei-peterson-publication-v10-*.tar.gz.sha256
```

El exportador verifica cada SHA-256 del manifest antes de comprimir y excluye `logs/` y
`fold_cache/`. El archivo completo original debe conservarse como procedencia interna.

## Extensión exploratoria Peterson Diffusion V11

V11 usa una rama y una carpeta independientes. No se ejecuta sobre el directorio de resultados V10
ni modifica sus 3,250 celdas. La primera corrida es la ola 0: 250 celdas `center_x6` destinadas a
compararse con `overlap` V10 bajo el mismo presupuesto.

```bash
cd /home/imiguel/Desktop/clei-2026-deep-eeg-benchmark
git fetch origin
git switch --track origin/exp/peterson-diffusion-v11
git pull --ff-only
conda activate deep-eeg-clei

tmux new -s peterson_diffusion_v11
export CLEI_DATA_DIR='/home/imiguel/Desktop/clei-2026-deep-eeg-benchmark/data/mi-openbci'
export DIFFUSION_WAVE='0'
export DIFFUSION_DEVICES='cuda:0 cuda:1'
export DIFFUSION_RESULTS_DIR="results_diffusion_v11/wave_00_$(git rev-parse --short HEAD)"
bash run_diffusion_cayetano.sh
```

Separar tmux sin detener la corrida: `Ctrl-b`, luego `d`. Volver a entrar:
`tmux attach -t peterson_diffusion_v11`.

Comprobar progreso:

```bash
pgrep -af 'run_diffusion_cayetano.sh|run_diffusion_v11.py'
find "$DIFFUSION_RESULTS_DIR/cells" -name '*.json' | wc -l
tail -n 30 "$DIFFUSION_RESULTS_DIR/logs/shard-0.log"
tail -n 30 "$DIFFUSION_RESULTS_DIR/logs/shard-1.log"
nvidia-smi
```

La ola 0 termina únicamente cuando `run_complete.json` existe y el auditor informa `250/250`:

```bash
python scripts/check_diffusion_v11.py \
  --output-dir "$DIFFUSION_RESULTS_DIR"
test -f "$DIFFUSION_RESULTS_DIR/run_complete.json" && echo 'wave_00=complete'
```

No lanzar ola 1 antes de transferir y analizar la ola 0 contra el resultado V10. El archivo
`decisions/wave_00.json` determina si `overlap` o `full` será la condición primaria siguiente.

## Búsqueda Peterson V12 — exp01

V12 no reemplaza V10/V11. Criba 36 configuraciones internas sobre seis sujetos de descubrimiento,
solo con validación interna `within_session` y ventanas `overlap`. Los cuatro sujetos holdout y el
test exterior no se leen ni se persisten durante `exp01`.

Actualizar y abrir tmux:

```bash
cd /home/imiguel/Desktop/clei-2026-deep-eeg-benchmark-v12
git fetch origin
git switch exp/peterson-model-search-v12
git pull --ff-only
conda activate deep-eeg-clei

tmux new -s peterson_search_v12
export CLEI_DATA_DIR='/home/imiguel/Desktop/clei-2026-deep-eeg-benchmark-v12/data/mi-openbci'
export PETERSON_SEARCH_EXPERIMENT='exp01'
export PETERSON_SEARCH_DEVICES='cuda:0 cuda:1'
export PETERSON_SEARCH_RESULTS_DIR="results_peterson_search_v12/exp01_$(git rev-parse --short HEAD)"
```

Primero ejecutar el smoke aislado (dos celdas, una por GPU, una época):

```bash
PETERSON_SEARCH_SMOKE=1 bash run_peterson_search_cayetano.sh
test -f "${PETERSON_SEARCH_RESULTS_DIR}_smoke/smoke_complete.json" && echo 'smoke=OK'
```

Solo si el smoke termina con `peterson_search_smoke=OK`, lanzar la corrida completa:

```bash
bash run_peterson_search_cayetano.sh
```

Separar tmux con `Ctrl-b`, luego `d`. Volver con `tmux attach -t peterson_search_v12`.

Monitorear:

```bash
pgrep -af 'run_peterson_search_cayetano.sh|run_peterson_search_v12.py'
find "$PETERSON_SEARCH_RESULTS_DIR/cells" -name '*.json' | wc -l
tail -n 30 "$PETERSON_SEARCH_RESULTS_DIR/logs/shard-0.log"
tail -n 30 "$PETERSON_SEARCH_RESULTS_DIR/logs/shard-1.log"
nvidia-smi
```

La corrida final debe llegar a 216 celdas y producir `run_complete.json`, `ranking-exp01.json` y
`decisions/exp01.json`. La decisión queda en estado `proposed`: no inicia `exp02` automáticamente.

```bash
python scripts/check_peterson_search_v12.py --output-dir "$PETERSON_SEARCH_RESULTS_DIR"
python scripts/analyze_peterson_search_v12.py --output-dir "$PETERSON_SEARCH_RESULTS_DIR"
test -f "$PETERSON_SEARCH_RESULTS_DIR/run_complete.json" && echo 'exp01=complete'
```
