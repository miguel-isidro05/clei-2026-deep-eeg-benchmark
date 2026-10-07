#!/usr/bin/env bash
set -euo pipefail

: "${PETERSON_RESULTS_DIR:?Set PETERSON_RESULTS_DIR to the completed Peterson result directory}"

python -u scripts/check_peterson_journal.py --output-dir "$PETERSON_RESULTS_DIR"
python -u scripts/run_peterson_statistics.py --results-dir "$PETERSON_RESULTS_DIR"
if [[ -n "${PETERSON_LATENCY_DEVICE:-}" ]]; then
  python -u scripts/profile_latency.py \
    --device "$PETERSON_LATENCY_DEVICE" \
    --datasets MI-OpenBCI \
    --models EEGNet FBCNet ShallowConvNet EEGConformer \
    --batch-sizes 1 64 \
    --output-dir "$PETERSON_RESULTS_DIR/latency"
fi
python -u scripts/generate_peterson_figures.py --results-dir "$PETERSON_RESULTS_DIR"
python -u scripts/generate_peterson_readout.py --results-dir "$PETERSON_RESULTS_DIR"

echo "peterson_finalize=OK output=$PETERSON_RESULTS_DIR"
