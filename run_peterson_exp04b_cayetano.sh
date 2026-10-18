#!/usr/bin/env bash
set -euo pipefail

: "${PETERSON_EXP04B_RESULTS_DIR:?Set PETERSON_EXP04B_RESULTS_DIR}"
: "${PETERSON_EXP04B_EXP03_DIR:?Set PETERSON_EXP04B_EXP03_DIR to corrected Exp03 results}"
: "${PETERSON_EXP04B_DEVICES:?Set PETERSON_EXP04B_DEVICES='cuda:0 cuda:1'}"
: "${CLEI_DATA_DIR:?Set CLEI_DATA_DIR to Peterson MI-OpenBCI}"

read -r -a devices <<< "$PETERSON_EXP04B_DEVICES"
if [[ "${#devices[@]}" -ne 2 || "${devices[0]}" != cuda:* || "${devices[1]}" != cuda:* ]]; then
  echo "PETERSON_EXP04B_DEVICES must contain exactly two CUDA devices" >&2
  exit 2
fi
if [[ "${devices[0]}" == "${devices[1]}" ]]; then
  echo "PETERSON_EXP04B_DEVICES must name distinct devices" >&2
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

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
output="$PETERSON_EXP04B_RESULTS_DIR"
smoke_args=()
if [[ "${PETERSON_EXP04B_SMOKE:-0}" == "1" ]]; then
  output="${PETERSON_EXP04B_RESULTS_DIR%/}_smoke"
  smoke_args=(--smoke)
elif [[ -n "$(git status --porcelain)" ]]; then
  echo "Full Exp04b requires a clean Git tree" >&2
  exit 2
fi
mkdir -p "$output/logs"

for wave in w1 w2 w3; do
  python -u scripts/run_peterson_exp04b_v12.py \
    --output-dir "$output" --exp03-dir "$PETERSON_EXP04B_EXP03_DIR" \
    --wave "$wave" --plan-only "${smoke_args[@]}"

  pids=()
  for shard in 0 1; do
    python -u scripts/run_peterson_exp04b_v12.py \
      --output-dir "$output" --exp03-dir "$PETERSON_EXP04B_EXP03_DIR" \
      --wave "$wave" --device "${devices[$shard]}" \
      --num-shards 2 --shard-index "$shard" "${smoke_args[@]}" \
      2>&1 | tee "$output/logs/${wave}-shard-${shard}.log" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "Exp04b $wave failed; inspect $output/failures and logs" >&2
    exit 1
  fi
  python -u scripts/check_peterson_exp04b_v12.py \
    --output-dir "$output" --wave "$wave" "${smoke_args[@]}"
  python -u scripts/analyze_peterson_exp04b_v12.py \
    --output-dir "$output" --wave "$wave"
done

python -u scripts/check_peterson_exp04b_v12.py \
  --output-dir "$output" --wave all "${smoke_args[@]}"
echo "peterson_exp04b=OK output=$output"

