#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.env"
  set +a
fi

resolve_path() {
  local maybe_relative="$1"
  if [[ "${maybe_relative}" = /* ]]; then
    printf "%s\n" "${maybe_relative}"
  else
    printf "%s\n" "${REPO_ROOT}/${maybe_relative#./}"
  fi
}

require_command() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
}

count_nii_files() {
  local folder="$1"
  if [[ ! -d "${folder}" ]]; then
    echo "0"
    return
  fi
  find "${folder}" -maxdepth 1 -type f \( -name "*.nii.gz" -o -name "*.nii" \) | wc -l | tr -d " "
}

sync_split() {
  local split_name="$1"
  local split_target="${DATASET_DIR}/${split_name}"
  local found=0

  mkdir -p "${split_target}"

  while IFS= read -r -d '' split_source; do
    found=1
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --ignore-existing \
        --include "*/" \
        --include "*.nii" \
        --include "*.nii.gz" \
        --exclude "*" \
        "${split_source}/" "${split_target}/"
    else
      while IFS= read -r -d '' nifti_file; do
        cp -n "${nifti_file}" "${split_target}/"
      done < <(find "${split_source}" -type f \( -name "*.nii.gz" -o -name "*.nii" \) -print0)
    fi
  done < <(find "${EXTRACT_ROOT}" -type d \( -iname "${split_name}" -o -iname "${split_name}_*" \) -print0)

  if [[ "${found}" -eq 0 ]]; then
    rmdir "${split_target}" 2>/dev/null || true
  fi
}

require_command curl
require_command unzip

DATASET_ROOT_DIR="$(resolve_path "${DATASET_ROOT_DIR:-./data/datasets}")"
DATASET_DIR="$(resolve_path "${DATASET_DIR:-./data/datasets/topBrain-2025}")"
DOWNLOAD_DIR="$(resolve_path "${DOWNLOAD_DIR:-${DATASET_ROOT_DIR}/download}")"
EXTRACT_ROOT="${DOWNLOAD_DIR}/topbrain_extracted"

mkdir -p "${DOWNLOAD_DIR}" "${EXTRACT_ROOT}" "${DATASET_DIR}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 [zenodo_zip_url ...]"
  echo "No URL provided: download both TopBrain records in the script defaults."
  exit 0
fi

if [[ "$#" -gt 0 ]]; then
  DATASET_URLS=("$@")
else
  DATASET_URLS=(
    "https://zenodo.org/records/16623496/files/TopBrain_Data_Release_Batch1_073025.zip?download=1"
    "https://zenodo.org/records/16878417/files/TopBrain_Data_Release_Batches1n2_081425.zip?download=1"
  )
fi

echo "Download directory: ${DOWNLOAD_DIR}"
echo "Prepared dataset directory: ${DATASET_DIR}"
echo

for dataset_url in "${DATASET_URLS[@]}"; do
  archive_name="$(basename "${dataset_url%%\?*}")"
  archive_path="${DOWNLOAD_DIR}/${archive_name}"
  archive_part="${archive_path}.part"
  extract_dir="${EXTRACT_ROOT}/${archive_name%.zip}"

  if [[ -s "${archive_path}" ]]; then
    echo "Using existing archive ${archive_name}"
  else
    echo "Downloading ${archive_name}"
    curl -fL --retry 5 --retry-delay 2 --continue-at - \
      "${dataset_url}" \
      -o "${archive_part}"
    mv "${archive_part}" "${archive_path}"
  fi

  echo "Extracting ${archive_name}"
  mkdir -p "${extract_dir}"
  unzip -oq "${archive_path}" -d "${extract_dir}"
done

sync_split "imagesTr"
sync_split "labelsTr"
sync_split "imagesTs"
sync_split "labelsTs"

METADATA_DIR="${DATASET_DIR}/metadata"
mkdir -p "${METADATA_DIR}"
while IFS= read -r -d '' metadata_file; do
  rel_path="${metadata_file#${EXTRACT_ROOT}/}"
  safe_name="${rel_path//\//__}"
  cp -n "${metadata_file}" "${METADATA_DIR}/${safe_name}"
done < <(find "${EXTRACT_ROOT}" -type f -iname "*.txt" -print0)

echo
echo "TopBrain dataset prepared:"
echo "  imagesTr: $(count_nii_files "${DATASET_DIR}/imagesTr") files"
echo "  labelsTr: $(count_nii_files "${DATASET_DIR}/labelsTr") files"
echo "  imagesTs: $(count_nii_files "${DATASET_DIR}/imagesTs") files"
echo "  labelsTs: $(count_nii_files "${DATASET_DIR}/labelsTs") files"
echo "  metadata: ${METADATA_DIR}"
