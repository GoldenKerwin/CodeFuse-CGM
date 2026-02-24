# data

数据处理脚本与数据资产目录。

## 代码文件
- `preprocess.py`: 图节点文本化、切片等预处理逻辑
- `encode.py`: 训练输入编码相关逻辑

## 子目录
- `raw/`: 原始下载分片（jsonl/jsonl.gz）
- `processed/`: 结构化产物（parquet、graphs、splits）
- `index/`: 邻接索引
- `meta/`: release、manifest、构建报告

## 训练所需关键文件
- 图目录（JSON）：通常为 `data/processed/cgm_graphs/`
- 训练样本：`data/processed/cgm_splits/train.jsonl`
- 验证样本：`data/processed/cgm_splits/valid.jsonl`

## 输入输出约定
- 输入：S2 下载分片与构建配置
- 输出：供 `train/train.py` 直接消费的图和 JSONL 样本

## Quick Verification

```bash
ls data/processed/cgm_splits/train.jsonl data/processed/cgm_splits/valid.jsonl
ls data/processed/cgm_graphs | head
```

## Related Docs
- [Back to root README](../README.md)
- [`docs/DATA_FORMAT.md`](../docs/DATA_FORMAT.md)
