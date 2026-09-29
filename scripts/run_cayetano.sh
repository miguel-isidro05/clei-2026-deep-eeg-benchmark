#!/usr/bin/env bash
set -Eeuo pipefail

RESULTS_ROOT="${DEEP_EEG_RESULTS_DIR:-results}"
mkdir -p "${RESULTS_ROOT}/logs"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${RESULTS_ROOT}/logs/paper-${RUN_STAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
set -x

finish_report() {
  status="$1"
  python scripts/write_run_report.py \
    --results-dir "${RESULTS_ROOT}" --status "${status}" --log-file "${LOG_FILE}" || true
}
trap 'finish_report failed' ERR

echo "run_started_utc=${RUN_STAMP}"
echo "git_revision=$(git rev-parse HEAD)"
python -u scripts/preflight.py --require-cuda --min-cuda-devices 2 \
  --quality-output-dir "${RESULTS_ROOT}/quality"
python -u scripts/audit_signal_quality.py \
  --datasets MI-OpenBCI Souza2023 \
  --output-dir "${RESULTS_ROOT}/quality"
python -u scripts/run_paper.py --phase all --plan-only --output-dir "${RESULTS_ROOT}"

python -u scripts/run_paper.py --phase peterson --device cuda:0 --output-dir "${RESULTS_ROOT}" &
peterson_gpu0=$!
python -u scripts/run_paper.py --phase souza --device cuda:1 --output-dir "${RESULTS_ROOT}" &
souza_gpu1=$!
primary_failed=0
wait "${peterson_gpu0}" || primary_failed=1
wait "${souza_gpu1}" || primary_failed=1
if [[ "${primary_failed}" -ne 0 ]]; then
  exit 1
fi

python -u scripts/run_paper.py --phase ica-sensitivity --num-shards 2 --shard-index 0 \
  --device cuda:0 --output-dir "${RESULTS_ROOT}" &
ica_gpu0=$!
python -u scripts/run_paper.py --phase ica-sensitivity --num-shards 2 --shard-index 1 \
  --device cuda:1 --output-dir "${RESULTS_ROOT}" &
ica_gpu1=$!
ica_failed=0
wait "${ica_gpu0}" || ica_failed=1
wait "${ica_gpu1}" || ica_failed=1
if [[ "${ica_failed}" -ne 0 ]]; then
  exit 1
fi

python -u scripts/profile_latency.py --device cuda:0 --output-dir "${RESULTS_ROOT}/latency"
python -u scripts/run_statistics.py --results-dir "${RESULTS_ROOT}"
python -u scripts/generate_figures.py --results-dir "${RESULTS_ROOT}"
python -u scripts/generate_manuscript_results.py --results-dir "${RESULTS_ROOT}"
finish_report completed
trap - ERR
echo "run_completed_utc=$(date -u +%Y%m%dT%H%M%SZ)"
echo "experiment_report=${RESULTS_ROOT}/EXPERIMENT_LOG.md"
