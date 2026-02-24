# launch

DeepSpeed 启动脚本目录。

## 文件
- `zero2.sh`: Zero-2 启动模板
- `zero3.sh`: Zero-3 启动模板

## 用法
先指定配置文件名（相对 `config/`）：

```bash
export TRAIN_CONFIG=pretrain_template.json
bash launch/zero2.sh
```

或：

```bash
export TRAIN_CONFIG=posttrain_template.json
bash launch/zero3.sh
```

## 注意事项
- 启动前确认 deepspeed 环境可用
- 多机场景需额外设置 `MASTER_ADDR`, `MASTER_PORT`, `RANK` 等变量

## Related Docs
- [Back to root README](../README.md)
- [`docs/TRAINING_GUIDE.md`](../docs/TRAINING_GUIDE.md)
