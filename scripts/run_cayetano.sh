#!/usr/bin/env bash
set -Eeuo pipefail

RESULTS_ROOT="${PETERSON_RESULTS_DIR:-results_peterson_journal_v10}"
mkdir -p "${RESULTS_ROOT}/logs"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${RESULTS_ROOT}/logs/peterson-${RUN_STAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
set -x

finish_report() {
  local status="$1"
  python scripts/write_run_report.py \
    --results-dir "${RESULTS_ROOT}" --status "${status}" --log-file "${LOG_FILE}" || true
}
trap 'finish_report failed' ERR

echo "run_started_utc=${RUN_STAMP}"
echo "git_revision=$(git rev-parse HEAD)"
echo "scope=MI-OpenBCI_Peterson_only"
python -u scripts/preflight.py --datasets MI-OpenBCI \
  --models FBCNet EEGNet ShallowConvNet EEGConformer \
  --require-cuda --min-cuda-devices 2 --quality-output-dir "${RESULTS_ROOT}/quality"
python -u scripts/audit_signal_quality.py --datasets MI-OpenBCI \
  --output-dir "${RESULTS_ROOT}/quality"
python -u scripts/audit_erd_ers.py --datasets MI-OpenBCI --output-dir "${RESULTS_ROOT}"
python -u scripts/check_low_cost_csp.py --dataset MI-OpenBCI --condition full \
  --output "${RESULTS_ROOT}/quality/csp-MI-OpenBCI-full.json"
python -u scripts/check_low_cost_csp.py --dataset MI-OpenBCI --condition center \
  --output "${RESULTS_ROOT}/quality/csp-MI-OpenBCI-center.json"
python -u scripts/run_peterson_journal.py --plan-only --output-dir "${RESULTS_ROOT}"

python -u scripts/run_peterson_journal.py --device cuda:0 --output-dir "${RESULTS_ROOT}" \
  --num-shards 2 --shard-index 0 &
gpu0=$!
python -u scripts/run_peterson_journal.py --device cuda:1 --output-dir "${RESULTS_ROOT}" \
  --num-shards 2 --shard-index 1 &
gpu1=$!
failed=0
wait "${gpu0}" || failed=1
wait "${gpu1}" || failed=1
[[ "${failed}" -eq 0 ]]

python -u scripts/check_peterson_journal.py --output-dir "${RESULTS_ROOT}"
python -u scripts/run_peterson_statistics.py --results-dir "${RESULTS_ROOT}"
python -u scripts/profile_latency.py --datasets MI-OpenBCI \
  --models FBCNet EEGNet ShallowConvNet EEGConformer --device cuda:0 --windowed-transfer \
  --output-dir "${RESULTS_ROOT}/latency"
finish_report completed
trap - ERR
echo "run_completed_utc=$(date -u +%Y%m%dT%H%M%SZ)"
echo "results=${RESULTS_ROOT}"
