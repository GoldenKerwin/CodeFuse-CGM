# 数据格式与任务说明（预训练 + 后训练）

本文档面向当前仓库代码（`train/train.py`、`modeling/cgm.py`、`data/preprocess.py`、`data/encode.py`）说明：

1. 目前代码支持哪些任务
2. 预训练和后训练分别需要什么输入数据
3. 每个字段在训练中如何被处理
4. 数据构造与质检建议

---

## 1. 训练流程总览

当前训练是统一入口：

```bash
python train/train.py --c <配置文件>
```

通过配置区分：

- 预训练：`config/pretrain_template.json`
- 后训练：`config/posttrain_template.json`

两阶段本质是同一套模型与数据读取逻辑，不同点主要在：

- `use_chat`（是否套 chat 模板）
- 学习率/训练步数
- 初始化权重来源（后训练一般加载预训练输出）

---

## 2. 当前代码支持的任务

### 2.1 任务标签集合（代码内置）

在 `train/train.py` 中定义了任务集合：

- `graph_query`
- `api`
- `issue_fix`
- `unit_test`
- `readme_summary`

### 2.2 实际训练模式

1. 单任务模式（最常用）
- 配置里 `task` 通常设为 `issue_fix`（也可换成上面任一标签）
- 样本里的 `task` 字段可统一写同一个值

2. 多任务模式（`task=mft`）
- 当配置 `task` 为 `mft` 时，`collate` 会读取每条样本的 `task`，映射为任务 ID
- 样本 `task` 必须是上面任务集合之一，否则会报错

说明：当前模板配置均使用单任务 `issue_fix`。

---

## 3. 输入数据总结构

训练输入分两类文件：

1. 样本文件（JSONL）
- `train_files`
- `valid_files`

2. 图文件（JSON）
- 由样本字段 `repo` 指向
- 目录由配置 `graph_dir` 决定

关系：每条 JSONL 样本会加载对应一张图，并与问答文本一起喂给模型。

---

## 4. 样本文件格式（JSONL）

每行一个样本，最小可用格式：

```json
{"repo":"repo_demo.json","language":"python","prompt":"...","answer":"...","task":"issue_fix"}
```

### 4.1 字段定义与处理

1. `repo`（必填）
- 图文件名（相对 `graph_dir`）或绝对路径
- 训练时通过 `getRawGraph` 加载
- 文件不存在会直接报错

2. `language`（必填）
- 目前有效值：`python` 或 `java`
- 会影响 `data/preprocess.py` 里节点文本化规则

3. `prompt`（必填）
- 用户输入文本
- `use_chat=false`：直接分词编码
- `use_chat=true`：通过 `CGMEncoder.dataToInput` 套模板编码

4. `answer`（必填）
- 监督目标文本
- `use_chat=false`：`answer + eos` 参与 loss
- `use_chat=true`：assistant 段按模板参与 loss

5. `task`（条件必填）
- 单任务：可固定同一个值
- 多任务（`task=mft`）：必须为任务集合中的合法标签

### 4.2 序列与 loss 处理

在 `collate_cgm` 中：

- `input_ids = prompt_ids + answer_ids`（非 chat）
- `loss_mask`：仅 answer 部分为 1
- `qa_mask`：仅 query 部分为 1（用于注意力拼接逻辑）
- 最终构造：
  - `x = input_ids[:-1]`
  - `y = input_ids[1:]`
  - `loss_mask = loss_mask[1:]`

---

## 5. 图文件格式（JSON）

最小结构：

```json
{
  "nodes": [...],
  "edges": [...]
}
```

### 5.1 nodes 要求

每个节点至少包含：

- `nodeType`
- `id` 或 `nodeId`

如果二者都没有，会报错。

### 5.2 edges 要求

每条边至少包含：

- `source`
- `target`

若 `source/target` 在节点表中找不到，对应边会被跳过（不会中断训练）。

### 5.3 nodeType 与字段建议

#### Python 图推荐

- `File`
  - 推荐字段：`filePath`、`fileName`、`text`
- `TextFile`
  - 推荐字段：`name`、`text`
- `PaperChunk`（引文网络预训练推荐）
  - 推荐字段：`text`、`encoder_text`、`orig_paper_id`、`chunk_index`、`chunk_count`
- `Class`
  - 推荐字段：`classType`、`className`、`comment`、`text`
- `Function`
  - 推荐字段：`header`、`name`、`comment`、`text`
- `Attribute`
  - 推荐字段：`attributeType`、`name`、`comment`、`text`
- `Lambda`
  - 推荐字段：`text`

#### Java 图推荐

- `File`
  - 推荐字段：`path`、`name`
- `TextFile`
  - 推荐字段：`name`、`text`
- `Class`
  - 推荐字段：`modifiers`、`name`、`comment`
- `Method`
  - 推荐字段：`className`、`methodName`、`signature`、`modifiers`、`comment`、`text`
- `Field`
  - 推荐字段：`modifiers`、`fieldType`、`name`、`comment`

注意：`nodeType` 不在处理器白名单时会抛错。

---

## 6. 图文本化与切片规则（影响预训练质量）

图节点进入模型前会经历：

1. 节点结构 -> 文本串（`getPythonSentence/getJavaSentence`）
2. 文本分词
3. 按 512 token 切片
4. 每个切片做 encoder embedding

这意味着：

- 节点文本越结构化，图语义越稳定
- 超长节点会被切成多个“子节点嵌入”
- 同一原节点的子切片会在邻接矩阵中互联
- 如果你在 `get_S2ORC` 阶段已导出 `PaperChunk`，则运行时不再做“按512切片”，而是直接按 chunk 节点编码

---

## 7. 预训练数据建议（面向引文网络）

### 7.1 目标

让模型学习“图结构 + 文本内容”的对齐，建立综述生成基础能力。

### 7.2 推荐配置

参考 `config/pretrain_template.json`：

- `use_chat=false`
- `task=issue_fix`（作为占位单任务标签）
- `use_adj=true`（启用图邻接）

### 7.3 推荐输入组织

1. 图（`repo` 指向）
- 文献主节点：`PaperChunk`（推荐）
  - 超过 512 token 的论文会在数据预处理阶段切为多个 chunk 节点
  - 每个 chunk 节点保留原论文引用关系扩展边 + 同论文 chunk 全连接边
- 章节/片段节点：`Function` 或 `Class`
  - `text = section block`
- 引文边：`source -> target`

2. 样本文本
- `prompt`：综述目标或主题描述
- `answer`：可用模板文本/弱监督文本（用于先对齐）

### 7.4 预训练字段处理建议

- `title`：去空白、去控制字符，必须非空
- `abstract/fulltext`：清洗后写入节点 `text`
- `year`：用于采样优先级（例如近 5 年）
- `license`：用于白名单过滤（如 `odc-by`）
- `citation`：保留解析成功的边，去自环，去悬挂边

---

## 8. 后训练数据建议（SFT 对齐阶段）

### 8.1 目标

在高质量监督上对齐输出风格、可读性和任务完成度。

### 8.2 推荐配置

参考 `config/posttrain_template.json`：

- `use_chat=true`
- 加载预训练权重（`pretrained_model_path` 指向预训练输出）
- 学习率通常低于预训练

### 8.3 输入格式

后训练仍使用同样 JSONL + 图文件格式。

区别在于：

- `answer` 必须是高质量目标输出（不再用占位文本）
- `prompt` 要贴近真实下游请求
- 图可继续使用预训练构建的引文网络图，也可叠加任务相关节点

### 8.4 后训练字段处理建议

- `prompt`：明确任务边界和输出格式要求
- `answer`：人工清洗、去模板噪声、统一风格
- `task`：单任务保持一致，或启用 `mft` 做混合任务训练

---

## 9. 与 get_S2ORC 流水线的对接

`run_pipeline.py build` 默认可导出：

- `data/processed/cgm_graphs/citation_subgraph_000.json` ...（多子图，每图一条训练数据）
- `data/processed/cgm_splits/train.jsonl`
- `data/processed/cgm_splits/valid.jsonl`
- `data/processed/cgm_splits/subgraph_export_meta.json`

导出的图节点文本默认是：

```text
title: <title>
abstract: <abstract>
```

导出的边是有向引用关系：`source -> target`。

子图复杂度由 `get_S2ORC/subgraph_config.json` 控制，关键字段：

- `num_subgraphs`
- `min_nodes_per_graph`
- `max_nodes_per_graph`
- `min_hops`
- `max_hops`

构图策略：扩展阶段使用自然邻接做 hop 扩展；落图时保留原始有向引用边，不人为改向。

可直接用于当前训练代码：

- `graph_dir = data/processed/cgm_graphs`
- `train_files = data/processed/cgm_splits/train.jsonl`
- `valid_files = data/processed/cgm_splits/valid.jsonl`

---

## 10. 数据质检清单（强烈建议）

训练前至少检查：

1. 样本字段完整性
- `repo/language/prompt/answer` 非空率

2. 图合法性
- `nodeType` 合法率
- `id/nodeId` 缺失率
- 边端点可解析率

3. 文本统计
- `prompt`/`answer` 平均长度
- 节点文本长度分布（防止大量空节点）

4. 图统计
- 平均出度、最大出度
- 孤立点比例

5. 训练可用性
- 随机抽样 10 条跑 `tests_smoke` 同类单步验证

---

## 11. 常见问题

1. 为什么同一份格式既能用于预训练也能用于后训练？
- 因为当前代码路径统一，差异主要由 `use_chat`、`answer` 质量和初始化权重决定。

2. 预训练一定要有人工高质量 answer 吗？
- 不必须。预训练可先用弱监督/模板占位对齐图文通路；后训练再用高质量答案做对齐。

3. 哪些字段是硬必需？
- 样本：`repo/language/prompt/answer`
- 图节点：`nodeType + id|nodeId`
- 图边：`source/target`

4. 当前训练 batch 有什么限制？
- 图前向目前按 `batch_size=1` 设计，建议通过 `gradient_accumulation_steps` 放大全局 batch。
