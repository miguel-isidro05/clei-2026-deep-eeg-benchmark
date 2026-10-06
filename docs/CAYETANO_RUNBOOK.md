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
export EXPORT_FILE="/home/imiguel/Desktop/clei-exports/clei-peterson-v10-$(git rev-parse --short HEAD)-$(date -u +%Y%m%d).tar.gz"

mkdir -p /home/imiguel/Desktop/clei-exports
tar -czf "$EXPORT_FILE" "$PETERSON_RESULTS_DIR"
cd /home/imiguel/Desktop/clei-exports
sha256sum "$(basename "$EXPORT_FILE")" > "$(basename "$EXPORT_FILE").sha256"
ls -lh "$(basename "$EXPORT_FILE")" "$(basename "$EXPORT_FILE").sha256"
```

En la Mac, desde la carpeta donde quieras recibirlo:

```bash
scp hinton_2_cayetano:/home/imiguel/Desktop/clei-exports/clei-peterson-v10-*.tar.gz .
scp hinton_2_cayetano:/home/imiguel/Desktop/clei-exports/clei-peterson-v10-*.tar.gz.sha256 .
shasum -a 256 -c clei-peterson-v10-*.tar.gz.sha256
```
