#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"

: "${PETERSON_RESULTS_DIR:?Set PETERSON_RESULTS_DIR to the completed Peterson result directory}"

python -u scripts/check_peterson_journal.py --output-dir "$PETERSON_RESULTS_DIR"
python -u scripts/run_peterson_statistics.py --results-dir "$PETERSON_RESULTS_DIR"
latency_devices="${PETERSON_LATENCY_DEVICES:-${PETERSON_LATENCY_DEVICE:-}}"
if [[ -n "$latency_devices" ]]; then
  read -r -a devices <<< "$latency_devices"
  if [[ "${#devices[@]}" -lt 2 ]]; then
    echo "PETERSON_LATENCY_DEVICES must contain at least two devices" >&2
    exit 2
  fi
  latency_parts="$PETERSON_RESULTS_DIR/latency_parts"
  mkdir -p "$latency_parts"
  pids=()
  inputs=()
  for device in "${devices[@]}"; do
    label="${device//:/_}"
    output="$latency_parts/$label"
    inputs+=("$output")
    python -u scripts/profile_latency.py \
      --device "$device" \
      --datasets MI-OpenBCI \
      --models EEGNet FBCNet ShallowConvNet EEGConformer \
      --batch-sizes 1 64 \
      --windowed-transfer \
      --output-dir "$output" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  python -u scripts/merge_latency_profiles.py \
    --inputs "${inputs[@]}" \
    --output-dir "$PETERSON_RESULTS_DIR/latency"
fi
python -u scripts/generate_peterson_figures.py --results-dir "$PETERSON_RESULTS_DIR"
python -u scripts/generate_peterson_readout.py --results-dir "$PETERSON_RESULTS_DIR"

echo "peterson_finalize=OK output=$PETERSON_RESULTS_DIR"
