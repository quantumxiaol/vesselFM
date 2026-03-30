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
TOPBRAIN_FINETUNE_DIR="$(resolve_path "${TOPBRAIN_FINETUNE_DIR:-${DATASET_DIR}/vesselfm_finetune}")"
HF_HOME="$(resolve_path "${HF_HOME:-./modelsweights}")"
PRETRAIN_CKPT="$(resolve_path "${PRETRAIN_CKPT:-${HF_HOME}/vesselFM_base.pt}")"
CHECKPOINTS_DIR="$(resolve_path "${CHECKPOINTS_DIR:-./checkpoints}")"
OUTPUTS_DIR="$(resolve_path "${OUTPUTS_DIR:-./outputs}")"

GPU_DEVICE="${GPU_DEVICE:-0}"
TRAIN_DEVICES="${TRAIN_DEVICES:-${GPU_DEVICE}}"
NUM_SHOTS="${NUM_SHOTS:-all}"
RUN_NAME="${RUN_NAME:-topbrain}"
WANDB_PROJECT="${WANDB_PROJECT:-vesselfm_topbrain}"
WANDB_OFFLINE="${WANDB_OFFLINE:-True}"
BATCH_SIZE="${BATCH_SIZE:-2}"
INPUT_SIZE="${INPUT_SIZE:-[128,128,128]}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
MAX_STEPS="${MAX_STEPS:-1200}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-200}"

normalize_devices_for_hydra() {
  local raw="${1// /}"
  if [[ -z "${raw}" ]]; then
    echo "[0]"
    return
  fi
  if [[ "${raw}" == \[*\] ]]; then
    echo "${raw}"
    return
  fi
  if [[ "${raw}" == *,* ]]; then
    echo "[${raw}]"
    return
  fi
  echo "[${raw}]"
}

HYDRA_DEVICES="$(normalize_devices_for_hydra "${TRAIN_DEVICES}")"
configure_venv_cuda_libs

mkdir -p "${CHECKPOINTS_DIR}" "${OUTPUTS_DIR}"

if [[ ! -d "${TOPBRAIN_FINETUNE_DIR}/train" || ! -d "${TOPBRAIN_FINETUNE_DIR}/val" || ! -d "${TOPBRAIN_FINETUNE_DIR}/test" ]]; then
  echo "Prepared split dataset not found, running prepare script..."
  python "${REPO_ROOT}/scripts/prepare_topbrain_finetune_data.py" \
    --dataset-dir "${DATASET_DIR}" \
    --output-dir "${TOPBRAIN_FINETUNE_DIR}" \
    --force
fi

if [[ "${NUM_SHOTS}" == "all" ]]; then
  NUM_SHOTS="$(find "${TOPBRAIN_FINETUNE_DIR}/train" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
fi

if [[ "${NUM_SHOTS}" -le 0 ]]; then
  echo "NUM_SHOTS must be > 0, got ${NUM_SHOTS}" >&2
  exit 1
fi

if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
  echo "Pretrained checkpoint not found: ${PRETRAIN_CKPT}" >&2
  exit 1
fi

export TOPBRAIN_FINETUNE_DIR
cd "${REPO_ROOT}"

echo "Starting finetune with:"
echo "  data path      : ${TOPBRAIN_FINETUNE_DIR}"
echo "  pretrained ckpt: ${PRETRAIN_CKPT}"
echo "  checkpoints    : ${CHECKPOINTS_DIR}"
echo "  num_shots      : ${NUM_SHOTS}"
echo "  train devices  : ${HYDRA_DEVICES}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
fi

python -m vesselfm.seg.finetune \
  data=eval_topbrain \
  num_shots="${NUM_SHOTS}" \
  devices="${HYDRA_DEVICES}" \
  path_to_chkpt="${PRETRAIN_CKPT}" \
  chkpt_folder="${CHECKPOINTS_DIR}" \
  run_name="${RUN_NAME}" \
  wandb_project="${WANDB_PROJECT}" \
  offline="${WANDB_OFFLINE}" \
  batch_size="${BATCH_SIZE}" \
  input_size="${INPUT_SIZE}" \
  dataloader.num_workers="${NUM_WORKERS}" \
  dataloader.prefetch_factor="${PREFETCH_FACTOR}" \
  trainer.lightning_trainer.max_steps="${MAX_STEPS}" \
  trainer.lightning_trainer.val_check_interval="${VAL_CHECK_INTERVAL}"
