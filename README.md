# CodeFuse-CGM

面向图增强代码生成模型（CGM）的训练与数据构建仓库。当前代码包含两条主线：
- 训练主线：`train/train.py` + `modeling/` + `data/` + `utils/`
- Semantic Scholar 数据主线：`run_pipeline.py` + `get_S2ORC/`

## Repository Guide
- [`config/README.md`](config/README.md): 训练与流水线配置模板
- [`data/README.md`](data/README.md): 数据格式、预处理脚本与目录约定
- [`docs/README.md`](docs/README.md): 使用文档入口
- [`get_S2ORC/README.md`](get_S2ORC/README.md): S2ORC 拉取与子图构建流水线
- [`train/README.md`](train/README.md): 训练入口与运行方式
- [`modeling/README.md`](modeling/README.md): CGM 模型结构说明
- [`utils/README.md`](utils/README.md): 参数、训练循环、loss 工具
- [`launch/README.md`](launch/README.md): DeepSpeed 启动脚本说明
- [`tests_smoke/README.md`](tests_smoke/README.md): 冒烟测试数据与执行方法
- [`encoder/README.md`](encoder/README.md): 本地编码器权重目录说明
- [`LLM/README.md`](LLM/README.md): 本地 LLM 权重目录说明
- [`models/README.md`](models/README.md): 自定义模型实现目录约定
- [`logs/README.md`](logs/README.md): 日志目录约定
- [`skills/README.md`](skills/README.md): 仓库内自定义 Codex skills

## 核心能力
1. 两阶段训练（预训练 + 后训练）
2. 基于图结构 + 文本的联合建模
3. S2 数据自动下载、过滤、索引与子图导出
4. 可选 DeepSpeed 启动（zero2 / zero3）

## 环境要求
- Python 3.11+
- 建议 Linux + CUDA 环境
- Semantic Scholar 数据构建需设置 `S2_API_KEY`

安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始
1. 准备训练数据格式（见 `docs/DATA_FORMAT.md`）
2. 运行最小冒烟：

```bash
python train/train.py --c tests_smoke/smoke_config.json
```

3. 运行预训练：

```bash
python train/train.py --c config/pretrain_template.json
```

4. 运行后训练：

```bash
python train/train.py --c config/posttrain_template.json
```

## Semantic Scholar 数据流水线
先设置 API Key：

```bash
export S2_API_KEY="<your_semantic_scholar_api_key>"
```

常用命令：

```bash
python run_pipeline.py list
python run_pipeline.py download --datasets citations --max-files 2
python run_pipeline.py build --target-papers 10000 --subgraph-config get_S2ORC/subgraph_config.json
```

生成目录：
- `data/raw/`, `data/processed/`, `data/index/`, `data/meta/`
- `logs/pipeline.log`

## 目录结构（顶层）

```text
CodeFuse-CGM/
├── run_pipeline.py
├── config/
├── data/
├── docs/
├── get_S2ORC/
├── train/
├── modeling/
├── utils/
├── launch/
├── tests_smoke/
├── encoder/
├── LLM/
├── models/
├── logs/
└── skills/
```

## Quick Verification
- `python -m py_compile run_pipeline.py`
- `python train/train.py --c tests_smoke/smoke_config.json`
- 产物检查：`tests_smoke/out/` 下出现 `final_step_1` checkpoint

## Troubleshooting
- 训练 OOM：降低 `max_len` / `max_decoder_length`，提高 `gradient_accumulation_steps`
- S2 API 失败：确认 `S2_API_KEY` 和网络可用
- 配置报错：先对照 `config/*.json` 模板和 `docs/TRAINING_GUIDE.md`

## Contributing
提交前建议：
1. 跑一次 `tests_smoke`
2. 更新对应目录 README（若行为或参数发生变化）
3. 避免提交模型权重、缓存和日志

## License
见 [`LEGAL.md`](LEGAL.md)。
