# models

模型兼容与自定义实现目录。

## 当前结构
- `qwen2/`: 自定义 Qwen2 相关实现（按版本子目录组织）

## 使用场景
当配置中开启 `self_defined=true` 时，训练流程可使用该目录下的自定义模型实现。

## 注意事项
- 与 transformers 版本强相关
- 升级依赖时需同步回归验证自定义实现

## Related Docs
- [Back to root README](../README.md)
