# modeling

CGM 主模型实现目录。

## 文件
- `cgm.py`: 图编码与语言模型融合的核心实现

## 角色
- 接收图结构编码结果与文本 token
- 组织注意力输入并计算训练所需前向输出

## 维护建议
- 修改前先确保 `train/train.py` 调用参数兼容
- 变更模型输入输出时同步更新 `docs/DATA_FORMAT.md`

## Related Docs
- [Back to root README](../README.md)
- [`train/README.md`](../train/README.md)
