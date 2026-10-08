#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -n "$(git status --porcelain --untracked-files=normal -- src scripts run_transfer_cayetano.sh)" ]]; then
  echo "Refusing transfer run with uncommitted scientific or launcher code." >&2
  exit 2
fi

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
OUTPUT_DIR="${TRANSFER_OUTPUT_DIR:-results_transfer_v9_repair}"
mkdir -p "$OUTPUT_DIR/logs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$OUTPUT_DIR/logs/transfer-$STAMP.log"
exec > >(tee -a "$LOG") 2>&1

wait_pair() {
  local first_pid="$1"
  local second_pid="$2"
  local failed=0
  wait "$first_pid" || failed=1
  wait "$second_pid" || failed=1
  if [[ "$failed" -ne 0 ]]; then
    return 1
  fi
}

echo "run_started_utc=$STAMP"
echo "git_revision=$(git rev-parse HEAD)"
python -u scripts/preflight.py --require-cuda --min-cuda-devices 2 --quality-output-dir "$OUTPUT_DIR/quality"
python -u scripts/audit_signal_quality.py --datasets MI-OpenBCI Souza2023 --output-dir "$OUTPUT_DIR/quality"
python -u scripts/audit_erd_ers.py --output-dir "$OUTPUT_DIR"
for DATASET in MI-OpenBCI Souza2023; do
  for CONDITION in full center; do
    python -u scripts/check_low_cost_csp.py \
      --dataset "$DATASET" \
      --condition "$CONDITION" \
      --output "$OUTPUT_DIR/quality/csp-${DATASET}-${CONDITION}.json"
  done
done

python -u scripts/run_transfer.py --phase pretrain --device cuda:0 --output-dir "$OUTPUT_DIR" --num-shards 2 --shard-index 0 &
SOURCE_0=$!
python -u scripts/run_transfer.py --phase pretrain --device cuda:1 --output-dir "$OUTPUT_DIR" --num-shards 2 --shard-index 1 &
SOURCE_1=$!
wait_pair "$SOURCE_0" "$SOURCE_1"
python -u scripts/check_transfer_v8.py --output-dir "$OUTPUT_DIR" --phase source

python -u scripts/run_peterson_v8.py --device cuda:0 --output-dir "$OUTPUT_DIR" --num-shards 2 --shard-index 0 &
PETERSON_0=$!
python -u scripts/run_peterson_v8.py --device cuda:1 --output-dir "$OUTPUT_DIR" --num-shards 2 --shard-index 1 &
PETERSON_1=$!
wait_pair "$PETERSON_0" "$PETERSON_1"

python -u scripts/run_transfer.py --phase target --device cuda:0 --output-dir "$OUTPUT_DIR" --num-shards 2 --shard-index 0 &
TARGET_0=$!
python -u scripts/run_transfer.py --phase target --device cuda:1 --output-dir "$OUTPUT_DIR" --num-shards 2 --shard-index 1 &
TARGET_1=$!
wait_pair "$TARGET_0" "$TARGET_1"

python -u scripts/check_transfer_v8.py --output-dir "$OUTPUT_DIR" --phase all
python -u scripts/run_transfer_statistics.py --output-dir "$OUTPUT_DIR"
python -u scripts/profile_latency.py --device cuda:0 --windowed-transfer \
  --output-dir "$OUTPUT_DIR/latency"
echo "run_completed_utc=$(date -u +%Y%m%dT%H%M%SZ)"
