# get_S2ORC 使用说明（Semantic Scholar 引文网络数据流水线）

本文档说明如何使用 `run_pipeline.py` + `get_S2ORC/` 模块，从 Semantic Scholar Datasets API 拉取数据并生成可用于本仓库训练的数据资产。
当前默认推荐使用方案A：`citations` 构结构 + Graph API 按 `CorpusId` 批量补齐 `title/abstract`（严格保证每个节点都有 `title` 和 `abstract`）。
导出阶段会把节点文本按编码器 tokenizer 切成 `<=512 token` 的 chunk，并扩展为 chunk 级图节点。

重要说明：
- `get_S2ORC` 不做 embedding。
- embedding 仍在训练阶段由 `train/train.py` 的 encoder 在线执行。

## 1. 功能概览

流水线分 3 个子命令：

1. `list`：查询 release 与 datasets 列表，写入 `data/meta/`
2. `download`：下载指定/自动筛选的数据分片到 `data/raw/`
3. `build`：流式解析并构建三表 + 图索引 +（可选）CGM 训练映射（多子图）

生成物：

- `data/processed/paper_nodes.parquet`
- `data/processed/citation_edges.parquet`
- `data/processed/paper_text_blocks.parquet`
- `data/index/adj_csr.npz + data/index/id_map.json` 或 `data/index/neighbors.parquet`
- 可选（默认开启）：
  - `data/processed/cgm_graphs/*.json`（每个子图 1 个文件）
  - `data/processed/cgm_splits/train.jsonl`
  - `data/processed/cgm_splits/valid.jsonl`
  - `data/processed/cgm_splits/subgraph_export_meta.json`

## 2. 环境要求

- Python 3.11+
- 已安装依赖：

```bash
conda run -n base python -m pip install -r requirements.txt
```

- 设置 API Key（推荐放 `.env`，项目已支持自动读取）：

```bash
S2_API_KEY=你的key
```

## 3. 目录与模块

- `run_pipeline.py`：CLI 入口
- `get_S2ORC/datasets_api.py`：Datasets API 客户端（限速、重试、缓存）
- `get_S2ORC/downloader.py`：预签名 URL 下载与断点续传
- `get_S2ORC/parser.py`：流式 JSONL/GZ 解析与字段标准化
- `get_S2ORC/filters.py`：质量过滤与级联清理
- `get_S2ORC/graph_index.py`：邻接索引构建
- `get_S2ORC/utils.py`：日志、文本处理、通用工具

## 4. API 与工程策略

### 4.1 使用的官方端点

- `GET /datasets/v1/release/`
- `GET /datasets/v1/release/{release_id}`
- `GET /datasets/v1/release/{release_id}/dataset/{dataset_name}`
- 可选：`GET /datasets/v1/diffs/{start_release_id}/to/{end_release_id}/{dataset_name}`

### 4.2 限速与重试

- 所有 Datasets API 请求共享全局限速：`1 request/second`
- 指数退避重试：最多 5 次
- 本地 API 缓存目录：`.cache/api/`

说明：获取到预签名文件 URL 后，文件下载不再算 API 请求。

## 5. 命令使用

### 5.1 查询 release

```bash
python run_pipeline.py list
```

输出（并落盘）：

- `data/meta/releases.json`
- `data/meta/latest_datasets.json`

### 5.2 下载分片（方案A推荐只下 `citations`）

自动筛选包含 `papers/abstracts/citation/reference` 关键词的数据集（兼容旧流程）：

```bash
python run_pipeline.py download --max-files 1
```

方案A推荐显式只下载 `citations`（更快、更省磁盘）：

```bash
conda run -n base python run_pipeline.py download --datasets citations --max-files 2
```

优先尝试 diff（不可用自动回退全量）：

```bash
python run_pipeline.py download --datasets papers --max-files 1 --prefer-diff
```

下载记录文件：`data/meta/download_manifest.json`

### 5.3 构建三表与索引

```bash
conda run -n base python run_pipeline.py build --target-papers 10000 --construction-mode graph_api_strict
```

常用参数：

- `--recent-years 5`：近年优先窗口
- `--min-year 2021`：直接指定最小年份
- `--construction-mode graph_api_strict|legacy`：构建模式（默认 `graph_api_strict`）
- `--allow-legacy-fallback`：允许 strict 失败后回退 legacy（默认关闭）
- `--max-citation-edges-scan`：citations 扫描边数上限（0=自动）
- `--license-allow odc-by,unknown`：许可白名单
- `--index-format csr|parquet_neighbors`
- `--direction outgoing|incoming|both`
- `--max-neighbors 50`
- `--no-export-cgm`：不导出训练映射
- `--subgraph-config get_S2ORC/subgraph_config.json`：子图构造配置（推荐）

## 6. 构建策略（方案A）

### 6.1 为什么默认不再依赖 `papers/abstracts` 分片 join

`papers` / `abstracts` / `citations` 的 bulk 分片在 shard 级别不保证按 `corpusid` 对齐。  
如果要求“每个节点必须同时有 `title` 和 `abstract`”，继续走跨分片 join 会掉量严重且浪费磁盘。

因此默认使用方案A：

1. 先扫描 `citations` 构造自然有向引用图（结构先行）
2. 根据 `subgraph_config.json` 采样候选子图（按 hop + 节点数控制复杂度）
3. 用 Graph API batch 按 `CorpusId` 批量补齐 `title/abstract`
4. 严格过滤掉缺 `title` 或缺 `abstract` 的节点
5. 导出训练图与样本

### 6.2 旧流程（legacy）

`legacy` 模式仍保留，但不推荐作为默认方案。它依赖 `papers/abstracts/citations` 的 bulk join，速度慢且命中率受 shard 错位影响大。

## 7. 字段标准化规则（build 阶段）

### 7.1 paper_id 生成优先级

1. `paperId`
2. `corpusId`
3. `doi`
4. `arXiv`
5. `pmid`
6. `sha1(title + year)`

### 7.2 三表字段

`paper_nodes`：

- `paper_id`
- `title`
- `abstract`
- `year`
- `venue`
- `fields_of_study`
- `doi`
- `arxiv_id`
- `pmid`
- `has_fulltext`
- `license`（缺失为 `unknown`）

`citation_edges`：

- `src_paper_id`
- `dst_paper_id`
- `is_resolved`
- `context`（最大 300 字符）

`paper_text_blocks`：

- `paper_id`
- `view_type`（`abstract|intro|method|conclusion|random_span`）
- `text`
- `token_len`
- `section_path`

### 7.3 质量过滤

- 仅保留 resolved 边：`src/dst` 都能在节点表中找到
- 仅保留可编码节点：`title` 非空 且 (`abstract` 或 `has_fulltext` 或 blocks>=1)
- 按 `--license-allow` 过滤许可
- 级联删边 + 删除孤立点
- 输出过滤前后统计到日志和 `data/meta/build_report.json`

## 8. 与当前训练代码的衔接

默认 `build` 会导出多子图训练映射（每个子图对应一条训练样本）：

- 图文件：`data/processed/cgm_graphs/citation_subgraph_000.json` ...（前缀可配）
- 样本文件：`data/processed/cgm_splits/train.jsonl`、`valid.jsonl`
- 导出统计：`data/processed/cgm_splits/subgraph_export_meta.json`

可直接映射到 `train/train.py` 所需输入格式（`graph/prompt/answer` + 图 JSON）。

其中图节点格式为（每个节点是一篇论文）：
其中图节点为 chunk 级节点：

- `nodeType=PaperChunk`
- `text`：可读文本（`title: ...` + `abstract: chunk片段`）
- `encoder_text`：训练编码实际使用文本（严格按 512-token chunk 生成）
- `orig_paper_id/chunk_index/chunk_count`：同一论文 chunk 追踪字段

边格式为自然有向引用边：`source -> target`（不做人为改向）。

chunk 连边规则：

- 原始边 `A->B` 扩展为 `A` 的所有 chunk 指向 `B` 的所有 chunk
- 同一论文多个 chunk 两两全连接（双向）

### 8.2 当前默认预训练问答任务（已切换）

当前导出的 `train.jsonl / valid.jsonl` 不再使用“占位综述答案”，而是默认生成：

- `prompt`：英文固定模板，要求模型根据 `[node_token_0]...[node_token_N]` 重构子图
- `answer`：纯 JSONL 文本（先 node 行，后 edge 行）

这与现有训练代码兼容（仍是 `prompt + answer` 自回归训练，`use_chat=false`）。

详细说明见：

- `get_S2ORC/PRETRAIN_RECONSTRUCTION.md`

一键启动预训练脚本（本地 `specter2_base` + `Qwen3-4B-Instruct-2`）：

- `get_S2ORC/run_pretrain_citation_reconstruct.sh`

### 8.1 子图复杂度配置（核心）

默认读取：`get_S2ORC/subgraph_config.json`，关键字段：

- `num_subgraphs`：子图数量（= 训练样本数）
- `min_nodes_per_graph`：每图最小节点数
- `max_nodes_per_graph`：每图最大节点数
- `min_hops`：最小 hop 扩展
- `max_hops`：最大 hop 扩展

说明：扩展时用“无向邻域”收集候选节点以获得结构复杂度，但子图落边仍保持原始有向引用关系。

## 9. 配置文件与一键启动

新增文件：

- `get_S2ORC/pipeline_config.json`：流水线总配置
- `get_S2ORC/subgraph_config.json`：子图复杂度配置
- `get_S2ORC/run_pipeline_from_config.sh`：一键执行脚本

一键运行：

```bash
bash get_S2ORC/run_pipeline_from_config.sh
```

说明：
- 脚本默认优先使用 `conda base`（`conda run --no-capture-output -n base python`）
- 可通过环境变量覆盖：`PY_BIN=/root/miniconda3/bin/python bash get_S2ORC/run_pipeline_from_config.sh`
- 脚本结束后会生成 `data/meta/train_ready_paths.json`，直接给训练配置填路径

指定配置运行：

```bash
bash get_S2ORC/run_pipeline_from_config.sh get_S2ORC/pipeline_config.json
```

## 10. 清理与整理（节省磁盘）

如果你已经尝试过旧流程、下载过大量 `papers/abstracts/s2orc` 分片，方案A下可以删除它们，只保留 `citations`：

```bash
find data/raw -maxdepth 1 -type f \\( -name 'papers__*.gz' -o -name 'abstracts__*.gz' -o -name 's2orc__*.gz' \\) -delete
```

重建前建议清理旧产物：

```bash
python - <<'PY'
from pathlib import Path
import shutil
for p in [Path('data/processed/cgm_graphs'), Path('data/processed/cgm_splits')]:
    shutil.rmtree(p, ignore_errors=True)
Path('data/processed/cgm_graphs').mkdir(parents=True, exist_ok=True)
Path('data/processed/cgm_splits').mkdir(parents=True, exist_ok=True)
for p in Path('data/processed').glob('*.parquet'):
    p.unlink(missing_ok=True)
for p in Path('data/index').glob('*'):
    if p.is_file():
        p.unlink(missing_ok=True)
PY
```

## 11. 故障排查

1. `Missing S2_API_KEY`
- 检查 `.env` 是否存在 `S2_API_KEY=...`
- 或在 shell 显式 `export S2_API_KEY=...`

2. 下载慢/失败
- 可能是网络波动，重试机制会自动生效
- 可降低并行动作（本实现默认串行）

3. `scipy` 未安装
- 若 `index-format=csr` 且本机缺 `scipy`，会自动回退 `parquet_neighbors`

4. `graph_api_strict` 构建失败或子图数量不足
- 常见原因：`citations` 覆盖不够（分片数太少）或 `subgraph_config` 更严格（如 `min_nodes` 提高）
- 优先增加 `citations` 分片数，而不是放松 `title/abstract` 硬约束
- 可增加 `--max-citation-edges-scan` 以提升结构候选覆盖
- 默认不会自动回退到 `legacy`。如需回退，显式开启 `--allow-legacy-fallback`

5. `ImportError: huggingface-hub ...`
- 原因：`transformers` 与 `huggingface-hub` 版本不兼容
- 当前仓库实测可用组合：`transformers==4.46.1` + `huggingface-hub==0.36.2`
- 修复示例：
  - `conda run -n base python -m pip install 'huggingface-hub>=0.23.2,<1.0' --upgrade`

## 12. 推荐最小可复现流程（方案A）

```bash
conda run -n base python run_pipeline.py list
conda run -n base python run_pipeline.py download --datasets citations --max-files 1
conda run -n base python run_pipeline.py build --target-papers 10000 --construction-mode graph_api_strict --index-format parquet_neighbors
```

## 13. 预训练一键启动与关键可控参数

你现在可以直接用以下脚本启动（默认读取 `config/cite_pretrain_template.json`）：

```bash
bash get_S2ORC/run_pretrain_with_template.sh
```

也可以指定启动方式：

```bash
LAUNCHER=python bash get_S2ORC/run_pretrain_with_template.sh
LAUNCHER=zero2  bash get_S2ORC/run_pretrain_with_template.sh
```

`config/cite_pretrain_template.json` 中建议重点关注这些可控项（无需改代码）：

- 显存/长度相关：
  - `graph_token_num`：每个样本最多保留多少图 token（已接入模型侧截断）
  - `seq_length`：QA token 截断长度（已接入 collate）
  - `mixed_precision`：`bf16|fp16|no`
  - `per_device_train_batch_size`
  - `gradient_accumulation_steps`
- 优化与训练步数：
  - `learning_rate`、`min_lr`、`weight_decay`、`lr_scheduler_type`
  - `num_train_epochs`、`max_train_steps`、`num_warmup_steps`
- 评估与保存：
  - `log_interval`
  - `step_evaluation`、`evaluation_steps`
  - `step_checkpointing`、`checkpointing_steps`
- 模型与LoRA：
  - `pretrained_encoder_path`、`pretrained_model_path`
  - `adapter_dtype`
  - `lora_rank/lora_alpha/lora_dropout`
  - `enc_lora_rank/enc_lora_alpha/enc_lora_dropout`

## Related Docs
- [Back to root README](../README.md)
- [PRETRAIN_RECONSTRUCTION](./PRETRAIN_RECONSTRUCTION.md)
