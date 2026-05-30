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
pip install -r requirements.txt
```

## Planned Workflow

1. Download and inspect DuReader_robust.
2. Verify data format and official evaluation script.
3. Build a minimal dev-set baseline.
4. Fine-tune one small pretrained QA model.
5. Compare accuracy, speed, and model size.
6. Summarize results for the course paper and presentation.
