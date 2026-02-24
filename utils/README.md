# utils

训练公共工具模块。

## 文件
- `arguments.py`: 参数解析与配置加载
- `common_utils.py`: 通用辅助函数
- `loss.py`: loss 计算相关逻辑
- `train_utils.py`: 训练循环与保存辅助

## 使用约定
- `train/train.py` 直接依赖本目录
- 修改工具函数签名时需要同步修改调用方

## Related Docs
- [Back to root README](../README.md)
- [`train/README.md`](../train/README.md)
