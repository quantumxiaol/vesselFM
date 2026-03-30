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

prepend_ld_library_path() {
  local path_to_add="$1"
  [[ -d "${path_to_add}" ]] || return
  if [[ -z "${LD_LIBRARY_PATH:-}" ]]; then
    LD_LIBRARY_PATH="${path_to_add}"
  elif [[ ":${LD_LIBRARY_PATH}:" != *":${path_to_add}:"* ]]; then
    LD_LIBRARY_PATH="${path_to_add}:${LD_LIBRARY_PATH}"
  fi
}

configure_venv_cuda_libs() {
  [[ -n "${VIRTUAL_ENV:-}" ]] || return
  local p
  for p in \
    "${VIRTUAL_ENV}"/lib/python*/site-packages/nvidia/nvjitlink/lib \
    "${VIRTUAL_ENV}"/lib/python*/site-packages/nvidia/cusparse/lib \
    "${VIRTUAL_ENV}"/lib/python*/site-packages/nvidia/cudnn/lib \
    "${VIRTUAL_ENV}"/lib/python*/site-packages/nvidia/cublas/lib \
    "${VIRTUAL_ENV}"/lib/python*/site-packages/nvidia/cuda_runtime/lib; do
    prepend_ld_library_path "${p}"
  done
  export LD_LIBRARY_PATH
}

DATASET_DIR="$(resolve_path "${DATASET_DIR:-./data/datasets/topBrain-2025}")"
CHECKPOINTS_DIR="$(resolve_path "${CHECKPOINTS_DIR:-./checkpoints}")"
OUTPUTS_DIR="$(resolve_path "${OUTPUTS_DIR:-./outputs}")"

GPU_DEVICE="${GPU_DEVICE:-0}"
INFER_DEVICE="${INFER_DEVICE:-${GPU_DEVICE}}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-4}"
INFER_PATCH_SIZE="${INFER_PATCH_SIZE:-[128,128,128]}"
INFER_OVERLAP="${INFER_OVERLAP:-0.5}"

configure_venv_cuda_libs

mkdir -p "${OUTPUTS_DIR}"

if [[ -z "${CKPT_PATH:-}" ]]; then
  CKPT_PATH="$(
    find "${CHECKPOINTS_DIR}" -type f -name "*.ckpt" -print0 \
      | xargs -0 ls -1t 2>/dev/null \
      | head -n 1
  )"
fi

if [[ -z "${CKPT_PATH:-}" ]]; then
  echo "No .ckpt file found under ${CHECKPOINTS_DIR}. Set CKPT_PATH manually." >&2
  exit 1
fi

CKPT_PATH="$(resolve_path "${CKPT_PATH}")"

if [[ -n "${IMAGE_PATH:-}" ]]; then
  IMAGE_PATH="$(resolve_path "${IMAGE_PATH}")"
elif [[ -d "${DATASET_DIR}/imagesTs" ]]; then
  IMAGE_PATH="${DATASET_DIR}/imagesTs"
else
  IMAGE_PATH="${DATASET_DIR}/imagesTr"
fi

if [[ -n "${MASK_PATH:-}" ]]; then
  MASK_PATH="$(resolve_path "${MASK_PATH}")"
  MASK_OVERRIDE="mask_path=${MASK_PATH}"
elif [[ -d "${DATASET_DIR}/labelsTs" ]]; then
  MASK_PATH="${DATASET_DIR}/labelsTs"
  MASK_OVERRIDE="mask_path=${MASK_PATH}"
else
  MASK_OVERRIDE="mask_path=null"
fi

PRED_DIR="$(resolve_path "${PRED_DIR:-${OUTPUTS_DIR}/topbrain_predictions}")"
mkdir -p "${PRED_DIR}"

if [[ "${INFER_DEVICE}" == *,* ]]; then
  echo "INFER_DEVICE must be a single device (e.g. 0 or cuda:0), got: ${INFER_DEVICE}" >&2
  exit 1
fi

if [[ "${INFER_DEVICE}" == cuda:* ]]; then
  DEVICE_ARG="${INFER_DEVICE}"
else
  DEVICE_ARG="cuda:${INFER_DEVICE}"
fi

echo "Running inference with:"
echo "  checkpoint : ${CKPT_PATH}"
echo "  image path : ${IMAGE_PATH}"
echo "  output path: ${PRED_DIR}"
echo "  device     : ${DEVICE_ARG}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
fi

python "${REPO_ROOT}/vesselfm/seg/inference.py" \
  ckpt_path="${CKPT_PATH}" \
  image_path="${IMAGE_PATH}" \
  output_folder="${PRED_DIR}" \
  device="${DEVICE_ARG}" \
  batch_size="${INFER_BATCH_SIZE}" \
  patch_size="${INFER_PATCH_SIZE}" \
  overlap="${INFER_OVERLAP}" \
  "${MASK_OVERRIDE}"
