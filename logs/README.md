# logs

运行日志目录。

## 当前文件
- `pipeline.log`: `run_pipeline.py` 执行日志

## 约定
- 训练日志与流水线日志默认写入该目录或其子目录
- 日志文件应被 `.gitignore` 忽略

## 常用排查

```bash
tail -n 200 logs/pipeline.log
```

## Related Docs
- [Back to root README](../README.md)
