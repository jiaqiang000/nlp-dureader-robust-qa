# 数据目录

原始数据和处理后的数据缓存不进入 Git。

下载后的目录规划：

```text
data/
├── raw/
│   └── dureader_robust-data/
└── processed/
```

`data/raw/` 用于保存官方 DuReader_robust 文件，`data/processed/` 用于保存后续可能生成的处理缓存。不要把大数据文件提交到仓库；数据下载地址和命令写在 README 或脚本中。

`scripts/inspect_dureader.py` 使用的官方数据包：

```text
https://bj.bcebos.com/paddlenlp/datasets/dureader_robust-data.tar.gz
```

下载的压缩包和解压后的数据都通过 `.gitignore` 中的 `data/raw/` 排除。
