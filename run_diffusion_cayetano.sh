#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root_dir"

: "${DIFFUSION_WAVE:?Set DIFFUSION_WAVE to 0 or 1}"
: "${DIFFUSION_RESULTS_DIR:?Set DIFFUSION_RESULTS_DIR to a new wave directory}"

read -r -a devices <<< "${DIFFUSION_DEVICES:-cuda:0 cuda:1}"
if [[ "${#devices[@]}" -ne 2 ]]; then
  echo "DIFFUSION_DEVICES must contain exactly two devices" >&2
  exit 2
fi

extra_args=()
smoke_mode="${DIFFUSION_SMOKE:-0}"
if [[ "$smoke_mode" == "1" ]]; then
  if [[ "$DIFFUSION_WAVE" != "1" ]]; then
    echo "DIFFUSION_SMOKE=1 is supported only for wave 1 GPU validation" >&2
    exit 2
  fi
  extra_args+=(--max-cells-per-shard 1 --epochs 1)
elif [[ "$smoke_mode" != "0" ]]; then
  echo "DIFFUSION_SMOKE must be 0 or 1" >&2
  exit 2
fi

export PYTHONPATH="$root_dir/src:$root_dir${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$DIFFUSION_RESULTS_DIR/logs"
started_utc="$(date -u +%Y%m%dT%H%M%SZ)"
main_log="$DIFFUSION_RESULTS_DIR/logs/wave-${DIFFUSION_WAVE}-${started_utc}.log"
exec > >(tee -a "$main_log") 2>&1

echo "run_started_utc=$started_utc"
echo "git_revision=$(git rev-parse HEAD)"
echo "wave=$DIFFUSION_WAVE devices=${devices[*]} output=$DIFFUSION_RESULTS_DIR"

python - "${devices[@]}" <<'PY'
import sys

import torch

from deepbench.config import MI_SUBJECTS, resolve_mi_data_dir

devices = sys.argv[1:]
for value in devices:
    target = torch.device(value)
    if target.type != "cuda":
        raise SystemExit(f"Cayetano V11 requires CUDA devices, got {value}")
    if not torch.cuda.is_available() or target.index is None or target.index >= torch.cuda.device_count():
        raise SystemExit(f"CUDA device unavailable: {value}")
data_dir = resolve_mi_data_dir()
missing = [subject for subject in MI_SUBJECTS if not (data_dir / f"{subject}.mat").exists()]
if missing:
    raise SystemExit(f"Missing Peterson files in {data_dir}: {missing}")
print(
    "preflight=OK "
    + " ".join(f"{value}={torch.cuda.get_device_name(torch.device(value))}" for value in devices),
    flush=True,
)
PY

python -u scripts/run_diffusion_v11.py \
  --wave "$DIFFUSION_WAVE" \
  --device "${devices[0]}" \
  --output-dir "$DIFFUSION_RESULTS_DIR" \
  --num-shards 2 --shard-index 0 \
  "${extra_args[@]}" \
  --plan-only

python -u scripts/run_diffusion_v11.py \
  --wave "$DIFFUSION_WAVE" \
  --device "${devices[0]}" \
  --output-dir "$DIFFUSION_RESULTS_DIR" \
  --num-shards 2 --shard-index 0 \
  "${extra_args[@]}" \
  > >(tee -a "$DIFFUSION_RESULTS_DIR/logs/shard-0.log") 2>&1 &
pid0=$!

python -u scripts/run_diffusion_v11.py \
  --wave "$DIFFUSION_WAVE" \
  --device "${devices[1]}" \
  --output-dir "$DIFFUSION_RESULTS_DIR" \
  --num-shards 2 --shard-index 1 \
  "${extra_args[@]}" \
  > >(tee -a "$DIFFUSION_RESULTS_DIR/logs/shard-1.log") 2>&1 &
pid1=$!

status0=0
status1=0
wait "$pid0" || status0=$?
wait "$pid1" || status1=$?
if [[ "$status0" -ne 0 || "$status1" -ne 0 ]]; then
  echo "diffusion_v11=failed shard0=$status0 shard1=$status1" >&2
  exit 1
fi

if [[ "$smoke_mode" == "1" ]]; then
  python -u scripts/check_diffusion_v11.py \
    --output-dir "$DIFFUSION_RESULTS_DIR" \
    --manifest-name smoke_expected.json
  echo "diffusion_v11_smoke=OK output=$DIFFUSION_RESULTS_DIR"
else
  python -u scripts/check_diffusion_v11.py \
    --output-dir "$DIFFUSION_RESULTS_DIR" \
    --write-complete
fi

echo "run_completed_utc=$(date -u +%Y%m%dT%H%M%SZ)"
echo "diffusion_v11=OK wave=$DIFFUSION_WAVE output=$DIFFUSION_RESULTS_DIR"
