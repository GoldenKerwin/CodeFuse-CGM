#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${ROOT_DIR}/config/cite_pretrain_template.json"
READY_MANIFEST="${ROOT_DIR}/data/meta/train_ready_paths.json"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export LAUNCHER="${LAUNCHER:-python}"   # python | zero2 | zero3

if [[ ! -f "${READY_MANIFEST}" ]]; then
  echo "[ERROR] 未找到 ${READY_MANIFEST}，请先执行 get_S2ORC/run_pipeline_from_config.sh 构建数据集。" >&2
  exit 1
fi

if [[ ! -f "${CFG}" ]]; then
  echo "[ERROR] 未找到训练配置 ${CFG}" >&2
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  PY="conda run --no-capture-output -n base python"
elif [[ -x "/root/miniconda3/bin/python" ]]; then
  PY="/root/miniconda3/bin/python"
else
  PY="python"
fi

# 用最新构建产物路径覆盖配置中的 graph/train/valid，避免手改配置。
TMP_CFG="${ROOT_DIR}/config/__tmp_cite_pretrain_with_data_$$.json"
cleanup() {
  rm -f "${TMP_CFG}" || true
}
trap cleanup EXIT

${PY} - <<'PY' "${CFG}" "${READY_MANIFEST}" "${TMP_CFG}"
import json, os, sys
cfg_path, manifest_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
m = json.load(open(manifest_path, "r", encoding="utf-8"))
for k in ("graph_dir", "train_files", "valid_files"):
    if k not in m or not m[k]:
        raise SystemExit(f"manifest缺少字段: {k}")
    if not os.path.exists(m[k]):
        raise SystemExit(f"manifest路径不存在: {k}={m[k]}")
cfg["graph_dir"] = m["graph_dir"]
cfg["train_files"] = m["train_files"]
cfg["valid_files"] = m["valid_files"]
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
print(out_path)
PY

CFG="${TMP_CFG}"

bash "${ROOT_DIR}/get_S2ORC/run_cite_pretrain.sh" "${CFG}"
