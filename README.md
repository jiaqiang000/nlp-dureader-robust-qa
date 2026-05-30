# DuReader Robust QA

This project is the experimental workspace for the NLP course final assignment.
The current topic is Chinese extractive question answering on DuReader_robust
with small pretrained language models under local low-compute constraints.

## Scope

- Task: extractive QA / machine reading comprehension.
- Dataset: DuReader_robust.
- Main metrics: F1, exact match, model size, training time, inference speed.
- Local priority: keep experiments runnable on the local Mac first.

## Project Rules

- Keep source code and experiment notes in Git.
- Do not commit raw datasets, processed caches, checkpoints, or prediction dumps.
- Start with scripts while the workflow is still exploratory.
- Add `src/`, `configs/`, or `tests/` only after code reuse or experiment scale makes them necessary.

## Environment

Create the virtual environment inside this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

For the current data-inspection stage, only the official evaluator dependency is
needed:

```bash
pip install six
```

Install the full ML dependency set before model fine-tuning:

```bash
pip install -r requirements.txt
```

## Data Inspection

Download DuReader_robust, inspect split statistics, and verify the official
evaluation script:

```bash
python scripts/inspect_dureader.py
```

The script downloads the official PaddleNLP data archive to `data/raw/`, which
is ignored by Git.

Observed statistics from the official archive:

| Split | Paragraphs | QA Pairs | Answers | Avg Context Chars | Avg Question Chars | Avg Answer Chars |
|---|---:|---:|---:|---:|---:|---:|
| train | 14,520 | 14,520 | 14,520 | 282.30 | 9.26 | 5.50 |
| dev | 1,417 | 1,417 | 1,962 | 284.28 | 9.42 | 6.45 |
| test | 31,032 | 50,000 | 0 | 304.38 | 10.29 | 0.00 |

The script also builds a dev-set prediction file from the first gold answer and
runs the official `evaluate.py`. Expected sanity-check result:

```json
{"F1": "100.000", "EM": "100.000", "TOTAL": 1417, "SKIP": 0}
```

Run unit tests:

```bash
python -m unittest discover -s tests -v
```

## Planned Workflow

1. Build a minimal dev-set baseline.
2. Fine-tune one small pretrained QA model.
3. Compare accuracy, speed, and model size.
4. Summarize results for the course paper and presentation.
