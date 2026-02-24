#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/get_S2ORC/pipeline_config.json}"
TARGET_OVERRIDE="${2:-}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "配置文件不存在: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  source "${ROOT_DIR}/.env"
  set +a
fi

if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS}" =~ ^[0-9]+$ || "${OMP_NUM_THREADS}" -lt 1 ]]; then
  export OMP_NUM_THREADS=1
fi

PY_BIN="${PY_BIN:-}"
if [[ -z "${PY_BIN}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    PY_BIN="conda run --no-capture-output -n base python"
  elif [[ -x "/root/miniconda3/bin/python" ]]; then
    PY_BIN="/root/miniconda3/bin/python"
  else
    PY_BIN="python"
  fi
fi

${PY_BIN} - <<'PY' "${ROOT_DIR}" "${CONFIG_PATH}" "${PY_BIN}" "${TARGET_OVERRIDE}"
import json
import os
import shlex
import subprocess
import sys

root = sys.argv[1]
cfg_path = sys.argv[2]
py_bin = sys.argv[3]
target_override = sys.argv[4].strip() if len(sys.argv) > 4 else ""

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

download = cfg.get("download", {})
build = cfg.get("build", {})
subgraph = cfg.get("subgraph", {})
paths_cfg = cfg.get("paths", {})

if target_override:
    subgraph["num_subgraphs"] = int(target_override)
elif os.getenv("TARGET_SUBGRAPHS", "").strip():
    subgraph["num_subgraphs"] = int(os.getenv("TARGET_SUBGRAPHS"))

runtime_subgraph_cfg = os.path.join(root, "data", "meta", "subgraph_runtime_config.json")
os.makedirs(os.path.dirname(runtime_subgraph_cfg), exist_ok=True)
with open(runtime_subgraph_cfg, "w", encoding="utf-8") as f:
    json.dump(subgraph, f, ensure_ascii=False, indent=2)


def run(cmd):
    print("+", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    if isinstance(py_bin, str) and py_bin.startswith("conda run "):
        prefix = shlex.split(py_bin)
    else:
        prefix = [py_bin]
    if cmd and cmd[0] == "python":
        cmd = prefix + cmd[1:]
    subprocess.run(cmd, cwd=root, check=True)

run(["python", "run_pipeline.py", "list"])

cmd_download = [
    "python", "run_pipeline.py", "download",
    "--max-files", str(download.get("max_files", 1)),
]
if download.get("datasets"):
    cmd_download += ["--datasets", str(download["datasets"])]
if download.get("prefer_diff", False):
    cmd_download += ["--prefer-diff"]
run(cmd_download)

cmd_build = [
    "python", "run_pipeline.py", "build",
    "--raw-dir", str(paths_cfg.get("raw_dir", "data/raw")),
    "--processed-dir", str(paths_cfg.get("processed_dir", "data/processed")),
    "--index-dir", str(paths_cfg.get("index_dir", "data/index")),
    "--target-papers", str(build.get("target_papers", 10000)),
    "--recent-years", str(build.get("recent_years", 5)),
    "--construction-mode", str(build.get("construction_mode", "graph_api_strict")),
    "--license-allow", str(build.get("license_allow", "odc-by,unknown")),
    "--index-format", str(build.get("index_format", "parquet_neighbors")),
    "--direction", str(build.get("direction", "outgoing")),
    "--max-neighbors", str(build.get("max_neighbors", 50)),
    "--valid-ratio", str(build.get("valid_ratio", 0.05)),
    "--subgraph-config", runtime_subgraph_cfg,
    "--resume-subgraphs",
    "--encoder-chunk-tokenizer-path", str(build.get("encoder_chunk_tokenizer_path", "/root/autodl-tmp/CodeFuse-CGM/encoder/specter2_base")),
]
if build.get("min_year") is not None:
    cmd_build += ["--min-year", str(build["min_year"])]
if build.get("max_citation_edges_scan"):
    cmd_build += ["--max-citation-edges-scan", str(build["max_citation_edges_scan"])]
if build.get("allow_legacy_fallback", False):
    cmd_build += ["--allow-legacy-fallback"]

run(cmd_build)

# Post-build validation + path manifest for training config
processed_dir = str(paths_cfg.get("processed_dir", "data/processed"))
graph_dir = os.path.join(processed_dir, "cgm_graphs")
split_dir = os.path.join(processed_dir, "cgm_splits")
train_path = os.path.join(split_dir, "train.jsonl")
valid_path = os.path.join(split_dir, "valid.jsonl")
assert os.path.exists(train_path), f"missing {train_path}"
assert os.path.exists(valid_path), f"missing {valid_path}"

with open(train_path, "r", encoding="utf-8") as f:
    first = json.loads(next(iter(f)).strip())
assert "prompt" in first and "answer" in first and "graph" in first

manifest = {
    "graph_dir": os.path.abspath(graph_dir),
    "train_files": os.path.abspath(train_path),
    "valid_files": os.path.abspath(valid_path),
    "paper_nodes_parquet": os.path.abspath(os.path.join(processed_dir, "paper_nodes.parquet")),
    "citation_edges_parquet": os.path.abspath(os.path.join(processed_dir, "citation_edges.parquet")),
    "paper_text_blocks_parquet": os.path.abspath(os.path.join(processed_dir, "paper_text_blocks.parquet")),
    "subgraph_stats": os.path.abspath(os.path.join(split_dir, "subgraph_export_meta.json")),
    "build_report": os.path.abspath(os.path.join(root, "data/meta/build_report.json")),
    "recommended_pretrain_config_template": os.path.abspath(os.path.join(root, "get_S2ORC/pretrain_citation_reconstruct_s2.json")),
}
manifest_path = os.path.join(root, "data", "meta", "train_ready_paths.json")
os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print("完成：已按配置生成多子图训练数据，并写出路径清单：", manifest_path)
PY
