#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

ENV_NAME="${CLEI_CONDA_ENV:-deep-eeg-clei}"
DATA_ROOT="${CLEI_DATA_ROOT:-${PROJECT_ROOT}/data}"
MI_DIR="${CLEI_DATA_DIR:-${DATA_ROOT}/mi-openbci}"
SOUZA_DIR="${SOUZA_DATA_DIR:-${DATA_ROOT}/souza2023}"
MNE_DIR="${MNE_DATA:-${DATA_ROOT}/mne}"
mkdir -p "${DATA_ROOT}" "${SOUZA_DIR}" "${SOUZA_DIR}/quarantine" "${MNE_DIR}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniconda before running setup.sh." >&2
  exit 1
fi
if ! command -v wget >/dev/null 2>&1; then
  echo "wget is required to download the EDF files." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda env create -n "${ENV_NAME}" -f environment.yml
else
  conda run -n "${ENV_NAME}" python -m pip install -e '.[dev]'
fi

if [[ ! -f "${MI_DIR}/S02.mat" ]]; then
  if [[ -e "${MI_DIR}" ]]; then
    echo "MI-OpenBCI directory exists but S02.mat is missing: ${MI_DIR}" >&2
    exit 1
  fi
  git clone --depth 1 https://github.com/ProMABLab/Database-MIOpenBCI.git "${MI_DIR}"
fi

while IFS=$'\t' read -r subject url expected_sha note; do
  [[ "${subject}" == "subject" ]] && continue
  if [[ "${subject}" == "001" && -z "${SOUZA_001_URL:-}" ]]; then
    target="${SOUZA_DIR}/quarantine/attachment-42-duplicate-004.edf"
  else
    [[ "${subject}" == "001" ]] && url="${SOUZA_001_URL}"
    target="${SOUZA_DIR}/${subject}.edf"
  fi
  if [[ ! -f "${target}" ]]; then
    wget --continue --output-document="${target}" "${url}"
  fi
  actual_sha="$(sha256sum "${target}" | awk '{print $1}')"
  if [[ "${subject}" != "001" || -z "${SOUZA_001_URL:-}" ]]; then
    if [[ "${actual_sha}" != "${expected_sha}" ]]; then
      echo "Checksum mismatch for ${target}: ${actual_sha}" >&2
      exit 1
    fi
  fi
  echo "souza_subject=${subject} sha256=${actual_sha} note=${note}"
done < configs/souza2023_downloads.tsv

export CLEI_DATA_DIR="${MI_DIR}"
export SOUZA_DATA_DIR="${SOUZA_DIR}"
export MNE_DATA="${MNE_DIR}"
conda env config vars set -n "${ENV_NAME}" \
  CLEI_DATA_DIR="${MI_DIR}" SOUZA_DATA_DIR="${SOUZA_DIR}" MNE_DATA="${MNE_DIR}"

validation_args=()
preflight_data_args=()
if [[ "${ALLOW_INCOMPLETE_SOUZA:-0}" == "1" ]]; then
  validation_args+=(--allow-incomplete)
  preflight_data_args+=(--allow-incomplete-souza)
fi
conda run -n "${ENV_NAME}" python scripts/validate_souza2023.py \
  --data-dir "${SOUZA_DIR}" "${validation_args[@]}"

conda run -n "${ENV_NAME}" python scripts/download_moabb.py \
  --datasets Zhou2020 Tavakolan2017

preflight_args=()
if [[ "${CAYETANO:-0}" == "1" ]]; then
  preflight_args+=(--require-cuda --min-cuda-devices 2)
fi
conda run -n "${ENV_NAME}" python scripts/preflight.py \
  "${preflight_args[@]}" "${preflight_data_args[@]}"

cat <<EOF
setup=OK
export CLEI_DATA_DIR='${MI_DIR}'
export SOUZA_DATA_DIR='${SOUZA_DIR}'
export MNE_DATA='${MNE_DIR}'
EOF
