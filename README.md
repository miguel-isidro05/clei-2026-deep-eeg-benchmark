# Deep EEG benchmark para CLEI 2026

Pipeline reproducible para cerrar la revision metodologica con MI-OpenBCI/Peterson solamente.
El perfil principal compara CSP+LDA, EEGNet, FBCNet, ShallowConvNet y EEGConformer en la tarea
motor imagery frente a rest.

Souza2023 y el codigo de transferencia quedan en el repositorio como trabajo historico y
exploratorio, pero no forman parte del runner confirmatorio actual.

## Alcance actual

- Dataset confirmatorio: MI-OpenBCI/Peterson, 10 sujetos, 15 canales, MI frente a rest.
- Modelos: CSP+LDA y cuatro redes profundas.
- Protocolos: split estratificado 70/30, five-fold within-session y LOSO.
- Seeds: 0, 1, 2, 3 y 4.
- Condicion primaria: seis ventanas solapadas de 2 s por trial, agregadas a una prediccion por trial.
- Controles: full trial, center crop, nonoverlap, center_x2 y center_x6.
- ICA: no ICA como principal; FastICA train-only con regla de kurtosis solo como sensibilidad.
- Total esperado: 3,250 celdas JSON.

Esta version no permite afirmar causalmente que el hardware low-cost sea mejor o peor que otro
hardware. La pregunta valida es: bajo un protocolo congelado dentro de Peterson, que decoder rinde
mejor y cuanto aporta la augmentacion por ventanas.

## Instalacion en Cayetano

```bash
git clone https://github.com/miguel-isidro05/clei-2026-deep-eeg-benchmark.git
cd clei-2026-deep-eeg-benchmark
git switch feat/peterson-journal-v10
CAYETANO=1 bash setup.sh
conda activate deep-eeg-clei
```

Si los MAT ya existen, puede fijar la ruta:

```bash
export CLEI_DATA_DIR=/home/imiguel/Desktop/clei-2026-deep-eeg-benchmark/data/mi-openbci
```

Souza2023 solo se descarga si se define `ENABLE_SOUZA=1`; el setup por defecto no lo necesita.

## Corrida completa

Ejecute dentro de `tmux`:

```bash
export PETERSON_RESULTS_DIR=results_peterson_journal_v10_$(git rev-parse --short HEAD)
bash run_cayetano.sh
```

El script usa dos GPU, guarda logs en `${PETERSON_RESULTS_DIR}/logs/`, valida 3,250 celdas, genera
tablas estadisticas y perfila latencia sobre entradas de 2 s, que son las usadas por la condicion
principal con sliding window. La corrida es reanudable: si se corta, repita el mismo comando con la
misma carpeta.

## Comprobacion

```bash
export PETERSON_RESULTS_DIR=results_peterson_journal_v10_$(git rev-parse --short HEAD)

pgrep -af 'run_cayetano.sh|run_peterson_journal.py' || echo "procesos=ninguno"
python scripts/check_peterson_journal.py --output-dir "$PETERSON_RESULTS_DIR"
test -f "$PETERSON_RESULTS_DIR/statistics/paired_wilcoxon_holm.csv" && echo "statistics=OK"
test -f "$PETERSON_RESULTS_DIR/latency/latency.json" && echo "latency=OK"
tail -n 40 "$(ls -1t "$PETERSON_RESULTS_DIR"/logs/peterson-*.log | head -n 1)"
```

No mezcle carpetas de resultados entre commits. Si cambia codigo, use una carpeta nueva.

## Salidas principales

- `cells/`: predicciones, metricas, seeds, hashes de indices y reportes de folds.
- `manifests/peterson-journal-expected.json`: manifiesto exacto del perfil.
- `statistics/descriptive_subject_seed_variability.csv`: accuracy y kappa por familia.
- `statistics/paired_wilcoxon_holm.csv`: comparaciones pareadas por sujeto.
- `statistics/augmentation_wilcoxon_holm.csv`: ventanas contra controles emparejados.
- `statistics/sample_accounting.csv`: trials, ejemplos, folds y actualizaciones.
- `statistics/expected_cell_audit.csv`: auditoria de celdas esperadas.
- `quality/`: calidad de senal, CSP sanity check y ERD/ERS.
- `latency/`: forward pass de los modelos profundos para entradas de 2 s.
- `logs/` y `EXPERIMENT_LOG.md`: consola y resumen de integridad.

## Verificacion local

```bash
conda run -n eeg-diffusion pytest tests/test_peterson_profile.py tests/test_classical.py \
  tests/test_protocols.py tests/test_statistics.py tests/test_cayetano_scope.py \
  tests/test_write_run_report.py -q
conda run -n eeg-diffusion ruff check .
```
