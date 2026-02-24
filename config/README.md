# config

训练与流水线配置模板目录。

## 包含文件
- `pretrain_template.json`: 预训练模板
- `posttrain_template.json`: 后训练模板
- `cite_pretrain_template.json`: 引文子图预训练模板
- `cite_pretrain_smoke_runtime.json`: 冒烟运行配置
- `cite_pretrain_zero2_smoke.json`: zero2 冒烟配置

## 常改字段
- 数据路径：`graph_dir`, `train_files`, `valid_files`
- 模型路径：`pretrained_encoder_path`, `pretrained_model_path`
- 输出路径：`output_dir`, `tb_dir`
- 训练超参：`lr`, `num_train_epochs`, `gradient_accumulation_steps`

## 使用方式

```bash
python train/train.py --c config/pretrain_template.json
python train/train.py --c config/posttrain_template.json
```

## Quick Verification
- JSON 格式合法（可用 `python -m json.tool` 检查）
- 配置里的路径在本机存在

## Related Docs
- [Back to root README](../README.md)
- [`docs/TRAINING_GUIDE.md`](../docs/TRAINING_GUIDE.md)
