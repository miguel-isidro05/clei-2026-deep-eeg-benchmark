#!/usr/bin/env bash
set -euo pipefail

: "${PETERSON_EXP03_RESULTS_DIR:?Set PETERSON_EXP03_RESULTS_DIR}"
: "${PETERSON_EXP03_DEVICES:?Set PETERSON_EXP03_DEVICES='cuda:0 cuda:1'}"
: "${CLEI_DATA_DIR:?Set CLEI_DATA_DIR to the Peterson MI-OpenBCI directory}"

read -r -a devices <<< "$PETERSON_EXP03_DEVICES"
if [[ "${#devices[@]}" -ne 2 || "${devices[0]}" != cuda:* || "${devices[1]}" != cuda:* ]]; then
  echo "PETERSON_EXP03_DEVICES must contain exactly two CUDA devices" >&2
  exit 2
fi
if [[ "${devices[0]}" == "${devices[1]}" ]]; then
  echo "PETERSON_EXP03_DEVICES must name distinct devices" >&2
  exit 2
fi

python - "${devices[@]}" <<'PY'
import sys
import torch

requested = [int(value.split(":", 1)[1]) for value in sys.argv[1:]]
available = torch.cuda.device_count()
if not torch.cuda.is_available() or available < 2:
    raise SystemExit(f"Two CUDA devices are required; detected={available}")
if any(index < 0 or index >= available for index in requested):
    raise SystemExit(f"Requested CUDA index outside available range: {requested}")
print(f"cuda_preflight=OK devices={requested}")
PY

for subject in S09 S03 S02 S10 S12 S08; do
  if [[ ! -f "$CLEI_DATA_DIR/$subject.mat" ]]; then
    echo "Missing Peterson discovery file: $CLEI_DATA_DIR/$subject.mat" >&2
    exit 2
  fi
done

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

smoke_args=()
output="$PETERSON_EXP03_RESULTS_DIR"
if [[ "${PETERSON_EXP03_SMOKE:-0}" == "1" ]]; then
  output="${PETERSON_EXP03_RESULTS_DIR%/}_smoke"
  smoke_args=(--smoke)
elif [[ -n "$(git status --porcelain)" ]]; then
  echo "Full exp03 requires a clean Git tree" >&2
  exit 2
fi

mkdir -p "$output/logs"

run_phase() {
  local phase="$1"
  python -u scripts/run_peterson_exp03_v12.py \
    --output-dir "$output" --phase "$phase" --plan-only "${smoke_args[@]}"

  python -u scripts/run_peterson_exp03_v12.py \
    --output-dir "$output" --phase "$phase" --device "${devices[0]}" \
    --num-shards 2 --shard-index 0 "${smoke_args[@]}" \
    2>&1 | tee "$output/logs/$phase-shard-0.log" &
  local pid0="$!"

  python -u scripts/run_peterson_exp03_v12.py \
    --output-dir "$output" --phase "$phase" --device "${devices[1]}" \
    --num-shards 2 --shard-index 1 "${smoke_args[@]}" \
    2>&1 | tee "$output/logs/$phase-shard-1.log" &
  local pid1="$!"

  local failed=0
  for pid in "$pid0" "$pid1"; do
    wait "$pid" || failed=1
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "At least one exp03 $phase shard failed; inspect $output/failures and logs" >&2
    exit 1
  fi

  python -u scripts/check_peterson_exp03_v12.py \
    --output-dir "$output" --phase "$phase" --mode strict "${smoke_args[@]}"
  python -u scripts/analyze_peterson_exp03_v12.py \
    --output-dir "$output" --phase "$phase" "${smoke_args[@]}"
}

run_phase recipe
run_phase quality

echo "peterson_exp03=OK output=$output"
