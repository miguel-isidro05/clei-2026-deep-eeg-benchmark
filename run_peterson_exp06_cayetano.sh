#!/usr/bin/env bash
set -euo pipefail

: "${PETERSON_EXP06_RESULTS_DIR:?Set PETERSON_EXP06_RESULTS_DIR}"
: "${PETERSON_EXP06_EXP05_DIR:?Set PETERSON_EXP06_EXP05_DIR to completed Exp05 output}"
: "${PETERSON_EXP06_DEVICES:?Set PETERSON_EXP06_DEVICES='cuda:0 cuda:1'}"
: "${CLEI_DATA_DIR:?Set CLEI_DATA_DIR to Peterson MI-OpenBCI}"

read -r -a devices <<< "$PETERSON_EXP06_DEVICES"
if [[ "${#devices[@]}" -ne 2 || "${devices[0]}" != cuda:* || "${devices[1]}" != cuda:* ]]; then
  echo "PETERSON_EXP06_DEVICES must contain exactly two CUDA devices" >&2
  exit 2
fi
if [[ "${devices[0]}" == "${devices[1]}" ]]; then
  echo "PETERSON_EXP06_DEVICES must name distinct devices" >&2
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
    raise SystemExit(f"CUDA index outside available range: {requested}")
print(f"cuda_preflight=OK devices={requested}")
PY

for subject in S09 S03 S02 S10 S12 S08; do
  if [[ ! -f "$CLEI_DATA_DIR/$subject.mat" ]]; then
    echo "Missing Peterson discovery file: $CLEI_DATA_DIR/$subject.mat" >&2
    exit 2
  fi
done
if [[ ! -f "$PETERSON_EXP06_EXP05_DIR/decisions/final-promotion.json" ]]; then
  echo "Missing Exp05 parent decision in $PETERSON_EXP06_EXP05_DIR" >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
output="$PETERSON_EXP06_RESULTS_DIR"
smoke_args=()
if [[ "${PETERSON_EXP06_SMOKE:-0}" == "1" ]]; then
  output="${PETERSON_EXP06_RESULTS_DIR%/}_smoke"
  smoke_args=(--smoke)
elif [[ -n "$(git status --porcelain)" ]]; then
  echo "Full Exp06 requires a clean Git tree" >&2
  exit 2
fi
mkdir -p "$output/logs"

run_phase() {
  local phase="$1"
  python -u scripts/run_peterson_exp06_v12.py \
    --output-dir "$output" --exp05-dir "$PETERSON_EXP06_EXP05_DIR" \
    --phase "$phase" --plan-only "${smoke_args[@]}"
  local pids=()
  for shard in 0 1; do
    python -u scripts/run_peterson_exp06_v12.py \
      --output-dir "$output" --exp05-dir "$PETERSON_EXP06_EXP05_DIR" \
      --phase "$phase" --device "${devices[$shard]}" \
      --num-shards 2 --shard-index "$shard" "${smoke_args[@]}" \
      2>&1 | tee "$output/logs/${phase}-shard-${shard}.log" &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "Exp06 $phase failed; inspect $output/logs" >&2
    exit 1
  fi
  python -u scripts/check_peterson_exp06_v12.py \
    --output-dir "$output" --exp05-dir "$PETERSON_EXP06_EXP05_DIR" \
    --phase "$phase" "${smoke_args[@]}"
  python -u scripts/analyze_peterson_exp06_v12.py \
    --output-dir "$output" --phase "$phase" "${smoke_args[@]}"
}

run_phase screen
run_phase confirm
python -u scripts/check_peterson_exp06_v12.py \
  --output-dir "$output" --exp05-dir "$PETERSON_EXP06_EXP05_DIR" \
  --phase all "${smoke_args[@]}"
echo "peterson_exp06=OK output=$output"
