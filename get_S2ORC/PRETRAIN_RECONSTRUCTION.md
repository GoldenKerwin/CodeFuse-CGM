# 引文网络重构预训练说明（S2 子图数据集）

本文档说明如何使用 `get_S2ORC` 生成的子图数据，构造并启动你当前预训练阶段的“引文网络重构”任务。

## 1. 任务定义（你当前的预训练任务）

目标不是节点补全，而是：

- 输入：图编码器产生的图 token（由子图节点文本编码得到）
- 输出：按固定 JSONL 格式重构该子图的节点与有向引用边

也就是“**图条件下的结构化文本重构生成**”。

说明：`get_S2ORC` 只负责把原始数据结构化成训练样本与图文件，不做 embedding。  
embedding 仍在训练时由 encoder 在线完成。

## 2. Prompt 模板（已写入导出的训练数据）

`data/processed/cgm_splits/train.jsonl` / `valid.jsonl` 中每条样本的 `prompt` 已按固定模板生成：

- `Input tokens: [node_token_0][node_token_1]...[node_token_N]`
- 要求先输出所有 node 行，再输出 edge 行
- 要求仅输出 JSONL 行，不要额外文本

注意：

- `node_token_i` 是占位符；训练时真正输入给模型的是图编码器输出的 graph embeddings（CGM 里在 attention 输入前拼接）。
- graph embedding 顺序与 `graphs/*.json` 的 `nodes` 顺序一致，因此与 answer 中 node 顺序一一对应。
- 超过 512 token 的节点会切 chunk 并复制图连边；每个 chunk 最终对应 1 个 graph token。

## 3. Answer 格式（已自动生成）

每条样本的 `answer` 是 JSONL 文本（多行字符串），格式为：

```json
{"type":"node","id":"paper_000","title":"...","abstract":"..."}
{"type":"node","id":"paper_001","title":"...","abstract":"..."}
{"type":"edge","source":"paper_001","target":"paper_000","relation":"cites"}
```

约束已满足：

- 所有 node 行在前
- 所有 edge 行在后
- `edge.source/target` 使用的 id 与前面的 node id 严格一致
- 无额外说明文字

## 4. 图文件格式（供现有训练代码读取）

图文件位于：`data/processed/cgm_graphs/*.json`

节点字段（训练代码可直接读取）：

- `id`
- `nodeType`（固定 `PaperChunk`）
- `text`（可读文本）
  - `title: ...`
  - `abstract: ...`（该 chunk 的摘要片段）
- `encoder_text`（训练编码实际使用文本）
- `orig_paper_id`、`chunk_index`、`chunk_count`

边字段：

- `source`
- `target`

边方向保持自然引用方向（`source -> target`）。

chunk 扩展规则：

- 单节点内容如果超过 512 token，会切成若干 chunk 节点
- 原始边 `A->B` 会扩展为所有 `A_chunk_i -> B_chunk_j`
- 同一论文的多个 chunk 两两全连接（双向）
- chunk 顺序由节点顺序 + decoder position embedding 维持

## 5. 本地模型路径（已写入预训练配置）

预训练配置文件：`get_S2ORC/pretrain_citation_reconstruct_s2.json`

已固定使用本地路径：

- 编码器：`/root/autodl-tmp/CodeFuse-CGM/encoder/specter2_base`
- LLM / tokenizer：`/root/autodl-tmp/CodeFuse-CGM/LLM/Qwen3-4B-Instruct-2`

关键说明：

- `specter2_base` 的 `hidden_size=768`
- 因此配置中的 `embedding_dim` 已设置为 `768`（否则适配器层会维度不匹配）

## 6. 启动预训练（一键）

脚本：`get_S2ORC/run_pretrain_citation_reconstruct.sh`

默认（单机直接 `python` 启动）：

```bash
bash get_S2ORC/run_pretrain_citation_reconstruct.sh
```

指定配置文件：

```bash
bash get_S2ORC/run_pretrain_citation_reconstruct.sh get_S2ORC/pretrain_citation_reconstruct_s2.json
```

脚本默认优先使用 `conda base`：

- `conda run --no-capture-output -n base python`

## 7. 使用现有 DeepSpeed 启动方式（策略不变）

如果你想继续使用现有 `launch/zero2.sh` / `launch/zero3.sh`：

```bash
LAUNCHER=zero2 bash get_S2ORC/run_pretrain_citation_reconstruct.sh
```

或：

```bash
LAUNCHER=zero3 bash get_S2ORC/run_pretrain_citation_reconstruct.sh
```

脚本会自动：

- 读取 `get_S2ORC/pretrain_citation_reconstruct_s2.json`
- 复制为临时 `config/*.json`
- 设置 `N_NODE/N_GPU_PER_NODE/RANK/MASTER_ADDR/MASTER_PORT`
- 调用现有 `launch/zero2.sh` 或 `launch/zero3.sh`

## 8. 重建数据（如果你修改了 `subgraph_config.json`）

```bash
conda run --no-capture-output -n base python run_pipeline.py build \
  --subgraph-config get_S2ORC/subgraph_config.json \
  --index-format parquet_neighbors
```

重建后会自动刷新：

- `data/processed/cgm_graphs/*.json`
- `data/processed/cgm_splits/train.jsonl`
- `data/processed/cgm_splits/valid.jsonl`

## 9. 快速自检（推荐）

1. 检查样本 `prompt` 中包含 `node_token_0`
2. 检查样本 `answer` 前几行是 `{"type":"node"...}`
3. 检查图文件节点 `text` 含 `title:` 和 `abstract:`
4. 用 `train/train.py` 的 `getRawGraph()` 读取任意一个 `graph` 路径

如果你后续要把任务从“重构全图 JSONL”切换成“仅预测边 / 仅预测节点 / 带描述边”，可以在 `run_pipeline.py` 的 `_export_cgm_prebuilt()` 中扩展模板与 `answer` 构造逻辑，而不需要改训练主流程。
