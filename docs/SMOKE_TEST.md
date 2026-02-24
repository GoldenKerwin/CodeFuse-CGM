# 最小闭环验证

本仓库内置样例：`tests_smoke/`

运行：

```bash
python train/train.py --c tests_smoke/smoke_config.json
```

成功标志：
- 打印训练日志
- 完成 1 step 更新
- 在 `tests_smoke/out/.../final_step_1` 下生成 checkpoint

该测试用于快速确认：
- 代码环境可用
- 数据格式可被读取
- 前向/反向/保存链路可闭环
