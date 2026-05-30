# Data Directory

Raw and processed datasets are intentionally excluded from Git.

Planned layout after download:

```text
data/
├── raw/
│   └── dureader_robust-data/
└── processed/
```

Use `data/raw/` for the official DuReader_robust files and `data/processed/`
for derived caches if needed. Keep dataset download links and commands in the
project README or scripts instead of committing large data files.
