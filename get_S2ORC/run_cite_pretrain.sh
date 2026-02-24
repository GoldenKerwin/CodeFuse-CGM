#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG_PATH="${1:-${ROOT_DIR}/config/cite_pretrain_template.json}"
LAUNCHER="${LAUNCHER:-python}"   # python | zero2 | zero3

if [[ ! -f "${CFG_PATH}" ]]; then
  echo "配置文件不存在: ${CFG_PATH}" >&2
  exit 1
fi

if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS}" =~ ^[0-9]+$ || "${OMP_NUM_THREADS}" -lt 1 ]]; then
  export OMP_NUM_THREADS=1
fi

if command -v conda >/dev/null 2>&1; then
  PY="conda run --no-capture-output -n base python"
  RUN_PREFIX="conda run --no-capture-output -n base"
elif [[ -x "/root/miniconda3/bin/python" ]]; then
  PY="/root/miniconda3/bin/python"
  RUN_PREFIX=""
else
  PY="python"
  RUN_PREFIX=""
fi

cd "${ROOT_DIR}"
echo "[INFO] 使用配置: ${CFG_PATH}"
echo "[INFO] 启动方式: ${LAUNCHER}"

if [[ "${LAUNCHER}" == "python" ]]; then
  echo "[INFO] 执行命令: ${PY} train/train.py --c ${CFG_PATH}"
  eval "${PY} train/train.py --c \"${CFG_PATH}\""
  exit 0
fi

if [[ "${LAUNCHER}" != "zero2" && "${LAUNCHER}" != "zero3" ]]; then
  echo "不支持的 LAUNCHER=${LAUNCHER}，可选: zero2|zero3|python" >&2
  exit 1
fi

TMP_CFG_NAME="__tmp_cite_pretrain_$(date +%s).json"
cp "${CFG_PATH}" "${ROOT_DIR}/config/${TMP_CFG_NAME}"

export N_NODE="${N_NODE:-1}"
if [[ -z "${N_GPU_PER_NODE:-}" ]]; then
  GPU_COUNT=$(eval "${PY} -c 'import torch; print(torch.cuda.device_count() or 1)'")
  export N_GPU_PER_NODE="${GPU_COUNT}"
fi
export RANK="${RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export TRAIN_CONFIG="${TMP_CFG_NAME}"

cleanup() {
  rm -f "${ROOT_DIR}/config/${TMP_CFG_NAME}" || true
}
trap cleanup EXIT

if [[ "${LAUNCHER}" == "zero2" ]]; then
  if [[ -n "${RUN_PREFIX}" ]]; then
    eval "${RUN_PREFIX} bash launch/zero2.sh"
  else
    bash launch/zero2.sh
  fi
else
  if [[ -n "${RUN_PREFIX}" ]]; then
    eval "${RUN_PREFIX} bash launch/zero3.sh"
  else
    bash launch/zero3.sh
  fi
fi
