#!/usr/bin/env bash
set -Eeuo pipefail

RESULTS_ROOT="${DEEP_EEG_RESULTS_DIR:-results_validation_no_loso_v7}"
VALIDATION_DEVICE="${VALIDATION_DEVICE:-cuda:0}"
mkdir -p "${RESULTS_ROOT}/logs"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${RESULTS_ROOT}/logs/validation-${RUN_STAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
set -x

finish_report() {
  exit_code="$1"
  status="failed"
  if [[ "${exit_code}" -eq 0 ]]; then
    status="completed"
  fi
  python scripts/write_run_report.py \
    --results-dir "${RESULTS_ROOT}" --status "${status}" --log-file "${LOG_FILE}" || true
}
trap 'finish_report "$?"' EXIT

echo "validation_started_utc=${RUN_STAMP}"
echo "git_revision=$(git rev-parse HEAD)"
echo "validation_role=complete_predeclared_no_loso_scope"
echo "validation_scope=both_datasets_all_blocks_except_loso"

python -u scripts/preflight.py --require-cuda --min-cuda-devices 1 \
  --quality-output-dir "${RESULTS_ROOT}/quality"
python -u scripts/audit_signal_quality.py \
  --datasets MI-OpenBCI Souza2023 --output-dir "${RESULTS_ROOT}/quality"
python -u scripts/check_low_cost_csp.py --dataset MI-OpenBCI \
  --output "${RESULTS_ROOT}/quality/peterson_csp_sanity.json"
python -u scripts/check_low_cost_csp.py --dataset Souza2023 \
  --output "${RESULTS_ROOT}/quality/souza_csp_sanity.json"

python -u scripts/run_paper.py --phase no-loso \
  --device "${VALIDATION_DEVICE}" --output-dir "${RESULTS_ROOT}"

python -u scripts/run_statistics.py --results-dir "${RESULTS_ROOT}"
python -u scripts/generate_figures.py --results-dir "${RESULTS_ROOT}"
python -u scripts/profile_latency.py \
  --device "${VALIDATION_DEVICE}" --output-dir "${RESULTS_ROOT}/latency"
echo "validation_completed_utc=$(date -u +%Y%m%dT%H%M%SZ)"
