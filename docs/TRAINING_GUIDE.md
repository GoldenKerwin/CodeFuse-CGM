# 训练指南（预训练 + 后训练）

## 1. 配置文件

- 预训练：`config/pretrain_template.json`
- 后训练：`config/posttrain_template.json`

你至少需要修改：
- `graph_dir`
- `train_files`
- `valid_files`
- `pretrained_encoder_path`
- `pretrained_model_path`
- `output_dir` / `tb_dir`

## 2. 两阶段建议

### 阶段1：预训练

目标：在引文网络通用数据上学习图-文本对齐与基础生成能力。

建议：
- `use_chat=false`
- `task=issue_fix`（或你自定义单任务）
- 学习率略高于后训练

### 阶段2：后训练

目标：在高质量综述指令数据上对齐输出风格与任务效果。

建议：
- `use_chat=true`
- 使用阶段1模型权重作为初始化
- 学习率小于预训练

## 3. 启动方式

### 单机直接启动

```bash
python train/train.py --c config/pretrain_template.json
python train/train.py --c config/posttrain_template.json
```

### DeepSpeed 启动

```bash
export TRAIN_CONFIG=pretrain_template.json
bash launch/zero2.sh
```

或：

```bash
export TRAIN_CONFIG=posttrain_template.json
bash launch/zero3.sh
```

## 4. 关键注意项

- 当前图训练前向默认按 `per_device_*_batch_size=1` 处理。
- 放大全局 batch 请使用 `gradient_accumulation_steps`。
- 若 `self_defined=true`，需准备自定义 Qwen2 对应依赖。
- 若 `self_defined=false`，推荐走标准 HF 模型路径。
