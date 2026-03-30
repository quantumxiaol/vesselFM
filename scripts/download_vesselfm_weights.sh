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

require_command curl

HF_HOME="$(resolve_path "${HF_HOME:-./modelsweights}")"
REPO_ID="${REPO_ID:-bwittmann/vesselFM}"
BASE_URL="https://huggingface.co/${REPO_ID}/resolve/main"

mkdir -p "${HF_HOME}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 [hf_file ...]"
  echo "No file provided: download vesselFM_base.pt and meta.yaml."
  exit 0
fi

if [[ "$#" -gt 0 ]]; then
  FILES=("$@")
else
  FILES=(
    "vesselFM_base.pt"
    "meta.yaml"
  )
fi

echo "Downloading VesselFM weights to: ${HF_HOME}"

for file_name in "${FILES[@]}"; do
  file_url="${BASE_URL}/${file_name}"
  file_path="${HF_HOME}/${file_name}"
  file_part="${file_path}.part"
  if [[ -s "${file_path}" ]]; then
    echo "Using existing ${file_name}"
  else
    echo "Downloading ${file_name}"
    curl -fL --retry 5 --retry-delay 2 --continue-at - \
      "${file_url}" \
      -o "${file_part}"
    mv "${file_part}" "${file_path}"
  fi
done

echo
echo "Done. Files:"
for file_name in "${FILES[@]}"; do
  ls -lh "${HF_HOME}/${file_name}"
done
