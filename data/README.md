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

Official data archive used by `scripts/inspect_dureader.py`:

```text
https://bj.bcebos.com/paddlenlp/datasets/dureader_robust-data.tar.gz
```

The downloaded archive and extracted files are ignored by Git through
`data/raw/`.
