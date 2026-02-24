# train

训练入口目录。

## 文件
- `train.py`: 主训练脚本（统一处理预训练与后训练）

## 基本命令

```bash
python train/train.py --c config/pretrain_template.json
python train/train.py --c config/posttrain_template.json
```

## 关键输入
- 配置文件：`--c <path/to/config.json>`
- 图文件目录：由配置 `graph_dir` 指定
- 样本文件：由配置 `train_files` / `valid_files` 指定

## 输出
- checkpoint: `output_dir`
- tensorboard: `tb_dir`

## Related Docs
- [Back to root README](../README.md)
- [`docs/TRAINING_GUIDE.md`](../docs/TRAINING_GUIDE.md)
