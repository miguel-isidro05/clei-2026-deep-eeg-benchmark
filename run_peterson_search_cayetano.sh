#!/usr/bin/env bash
set -euo pipefail

: "${PETERSON_SEARCH_EXPERIMENT:?Set PETERSON_SEARCH_EXPERIMENT=exp01}"
: "${PETERSON_SEARCH_RESULTS_DIR:?Set PETERSON_SEARCH_RESULTS_DIR}"
: "${PETERSON_SEARCH_DEVICES:?Set PETERSON_SEARCH_DEVICES='cuda:0 cuda:1'}"

if [[ "$PETERSON_SEARCH_EXPERIMENT" != "exp01" ]]; then
  echo "Only exp01 is implemented; later stages require an audited decision" >&2
  exit 2
fi

read -r -a devices <<< "$PETERSON_SEARCH_DEVICES"
if [[ "${#devices[@]}" -ne 2 || "${devices[0]}" != cuda:* || "${devices[1]}" != cuda:* ]]; then
  echo "PETERSON_SEARCH_DEVICES must contain exactly two CUDA devices" >&2
  exit 2
fi
if [[ "${devices[0]}" == "${devices[1]}" ]]; then
  echo "PETERSON_SEARCH_DEVICES must name two distinct CUDA devices" >&2
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

: "${CLEI_DATA_DIR:?Set CLEI_DATA_DIR to the Peterson MI-OpenBCI directory}"
for subject in S09 S03 S02 S10 S12 S08; do
  if [[ ! -f "$CLEI_DATA_DIR/$subject.mat" ]]; then
    echo "Missing Peterson discovery file: $CLEI_DATA_DIR/$subject.mat" >&2
    exit 2
  fi
done
echo "peterson_data_preflight=OK subjects=6"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

if [[ "${PETERSON_SEARCH_SMOKE:-0}" != "1" ]] && [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Full search requires a clean tracked Git tree" >&2
  exit 2
fi

output="$PETERSON_SEARCH_RESULTS_DIR"
smoke_args=()
if [[ "${PETERSON_SEARCH_SMOKE:-0}" == "1" ]]; then
  output="${PETERSON_SEARCH_RESULTS_DIR%/}_smoke"
  smoke_args=(--smoke --epochs-override 1 --max-cells-per-shard 1)
fi
mkdir -p "$output/logs"

python -u scripts/run_peterson_search_v12.py --output-dir "$output" --plan-only

python -u scripts/run_peterson_search_v12.py \
  --output-dir "$output" --device "${devices[0]}" \
  --num-shards 2 --shard-index 0 "${smoke_args[@]}" \
  2>&1 | tee "$output/logs/shard-0.log" &
pid0="$!"

python -u scripts/run_peterson_search_v12.py \
  --output-dir "$output" --device "${devices[1]}" \
  --num-shards 2 --shard-index 1 "${smoke_args[@]}" \
  2>&1 | tee "$output/logs/shard-1.log" &
pid1="$!"

failed=0
for pid in "$pid0" "$pid1"; do
  wait "$pid" || failed=1
done
if [[ "$failed" -ne 0 ]]; then
  echo "At least one V12 shard failed" >&2
  exit 1
fi

if [[ "${PETERSON_SEARCH_SMOKE:-0}" == "1" ]]; then
  python -u scripts/check_peterson_search_v12.py --output-dir "$output" --smoke
  echo "peterson_search_smoke=OK output=$output"
  exit 0
fi

python -u scripts/check_peterson_search_v12.py --output-dir "$output"
python -u scripts/analyze_peterson_search_v12.py --output-dir "$output"
echo "peterson_search_exp01=OK output=$output"
