# tests_smoke

最小闭环测试目录，用于快速验证环境与训练链路。

## 包含内容
- `smoke_config.json`: 冒烟配置
- `graphs/`: 示例图
- `splits/train.jsonl`, `splits/valid.jsonl`: 示例样本
- `out/`, `tb/`: 冒烟输出目录

## 执行

```bash
python train/train.py --c tests_smoke/smoke_config.json
```

## 成功标志
- 训练至少完成 1 step
- 生成 `tests_smoke/out/.../final_step_1` checkpoint

## Related Docs
- [Back to root README](../README.md)
- [`docs/SMOKE_TEST.md`](../docs/SMOKE_TEST.md)
