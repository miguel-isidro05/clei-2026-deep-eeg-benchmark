#!/usr/bin/env bash
set -Eeuo pipefail

RESULTS_ROOT="${DEEP_EEG_RESULTS_DIR:-results_validation_no_loso_v7}"
VALIDATION_SEED="${VALIDATION_SEED:-0}"
VALIDATION_EPOCHS="${VALIDATION_EPOCHS:-300}"
VALIDATION_SCOPE="${VALIDATION_SCOPE:-peterson}"
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

if [[ "${VALIDATION_SCOPE}" != "peterson" && "${VALIDATION_SCOPE}" != "both" ]]; then
  echo "VALIDATION_SCOPE must be peterson or both" >&2
  exit 2
fi

echo "validation_started_utc=${RUN_STAMP}"
echo "git_revision=$(git rev-parse HEAD)"
echo "validation_role=diagnostic_not_confirmatory"
echo "validation_scope=${VALIDATION_SCOPE}"
echo "validation_seed=${VALIDATION_SEED}"

python -u scripts/preflight.py --require-cuda --min-cuda-devices 1 \
  --quality-output-dir "${RESULTS_ROOT}/quality"
python -u scripts/check_peterson_csp.py \
  --output "${RESULTS_ROOT}/quality/peterson_csp_sanity.json"

python -u scripts/run_experiments.py \
  --dataset MI-OpenBCI \
  --models EEGNet FBCNet ShallowConvNet EEGConformer EEGInceptionMI \
  --protocols within_split within_session \
  --conditions full \
  --seeds "${VALIDATION_SEED}" \
  --epochs "${VALIDATION_EPOCHS}" \
  --ica-policy none \
  --device "${VALIDATION_DEVICE}" \
  --output-dir "${RESULTS_ROOT}"

if [[ "${VALIDATION_SCOPE}" == "both" ]]; then
  python -u scripts/run_experiments.py \
    --dataset Souza2023 \
    --models EEGNet FBCNet ShallowConvNet EEGConformer EEGInceptionMI \
    --protocols within_split within_session cross_session \
    --conditions full \
    --seeds "${VALIDATION_SEED}" \
    --epochs "${VALIDATION_EPOCHS}" \
    --ica-policy none \
    --device "${VALIDATION_DEVICE}" \
    --output-dir "${RESULTS_ROOT}"
fi

python -u scripts/run_statistics.py \
  --results-dir "${RESULTS_ROOT}" --seeds "${VALIDATION_SEED}" --allow-incomplete
echo "validation_completed_utc=$(date -u +%Y%m%dT%H%M%SZ)"
