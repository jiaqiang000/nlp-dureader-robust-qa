# Minimal QA Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tiny end-to-end DuReader_robust QA model path that trains on a small subset, predicts on a small dev subset, and evaluates with the official F1/EM script.

**Architecture:** Keep the project lightweight by adding one script under `scripts/` and focused tests under `tests/`. The script will reuse the existing downloaded DuReader_robust files, write temporary predictions and subset references under `outputs/minimal_qa/`, and keep all model/data artifacts out of Git.

**Tech Stack:** Python 3.9, PyTorch, HuggingFace Transformers, official DuReader_robust `evaluate.py`, standard-library `unittest`.

---

### Task 1: Test Pure QA Utilities

**Files:**
- Create: `tests/test_minimal_qa.py`
- Create later: `scripts/minimal_qa.py`

- [x] **Step 1: Write failing tests**

```python
import importlib.util
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "minimal_qa.py"
spec = importlib.util.spec_from_file_location("minimal_qa", SCRIPT_PATH)
minimal_qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(minimal_qa)


class MinimalQaTest(unittest.TestCase):
    def test_locate_answer_token_span_inside_context_window(self):
        offsets = [(0, 0), (0, 2), (0, 0), (0, 2), (2, 5), (5, 8), (0, 0)]
        sequence_ids = [None, 0, None, 1, 1, 1, None]

        span = minimal_qa.locate_answer_token_span(
            offsets=offsets,
            sequence_ids=sequence_ids,
            answer_start=2,
            answer_text="全称大",
            cls_index=0,
        )

        self.assertEqual(span, (4, 4))

    def test_locate_answer_token_span_returns_cls_when_answer_missing(self):
        offsets = [(0, 0), (0, 2), (0, 0), (0, 2), (2, 5), (0, 0)]
        sequence_ids = [None, 0, None, 1, 1, None]

        span = minimal_qa.locate_answer_token_span(
            offsets=offsets,
            sequence_ids=sequence_ids,
            answer_start=20,
            answer_text="不存在",
            cls_index=0,
        )

        self.assertEqual(span, (0, 0))

    def test_pick_best_answer_uses_offsets(self):
        context = "韩国全称大韩民国。"
        features = [
            {
                "example_id": "q1",
                "offset_mapping": [(0, 0), (0, 2), (2, 4), (4, 8), (8, 9)],
                "start_logits": [0.0, 0.1, 0.2, 5.0, 0.0],
                "end_logits": [0.0, 0.1, 0.2, 5.0, 0.0],
            }
        ]

        answer = minimal_qa.pick_best_answer(context, features)

        self.assertEqual(answer, "大韩民国")
```

- [x] **Step 2: Run test and verify failure**

Run: `.venv/bin/python -m unittest tests/test_minimal_qa.py -v`

Expected: fail because `scripts/minimal_qa.py` does not exist.

### Task 2: Implement Minimal QA Script

**Files:**
- Create: `scripts/minimal_qa.py`
- Modify: `requirements.txt`

- [x] **Step 1: Implement pure helpers**

Create `locate_answer_token_span`, `pick_best_answer`, dataset subsetting, and official-evaluator invocation. Keep heavy imports such as `torch` and `transformers` inside functions that need them so pure unit tests remain fast.

- [x] **Step 2: Run unit tests**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: all tests pass.

### Task 3: Run Tiny End-to-End Chain

**Files:**
- Runtime output only: `outputs/minimal_qa/`
- Modify: `README.md`

- [x] **Step 1: Install full dependencies**

Run: `.venv/bin/pip install -r requirements.txt`

Expected: PyTorch, Transformers, Datasets, Accelerate, and evaluator dependency install into `.venv`.

- [x] **Step 2: Run tiny chain**

Run:

```bash
.venv/bin/python scripts/minimal_qa.py \
  --train-samples 80 \
  --eval-samples 30 \
  --epochs 1 \
  --batch-size 8 \
  --max-length 256 \
  --seed 42 \
  --output-dir outputs/minimal_qa
```

Expected: `outputs/minimal_qa/predictions.json`, `outputs/minimal_qa/dev_subset.json`, and `outputs/minimal_qa/metrics.json` are created, and the official evaluator returns F1/EM for 30 dev examples.

### Task 4: Document and Publish

**Files:**
- Modify: `README.md`
- Modify: `scripts/README.md`

- [x] **Step 1: Document command and caution**

Record that the tiny run is a chain verification, not a formal paper result.

- [x] **Step 2: Verify before commit**

Run:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m py_compile scripts/inspect_dureader.py scripts/minimal_qa.py tests/test_inspect_dureader.py tests/test_minimal_qa.py
git status --short --ignored
```

Expected: tests pass, compile succeeds, `.venv/`, `data/raw/`, and `outputs/` are ignored.

- [x] **Step 3: Commit and push**

Run:

```bash
git add README.md scripts/README.md requirements.txt scripts/minimal_qa.py tests/test_minimal_qa.py docs/superpowers/plans/2026-05-31-minimal-qa-chain.md
git commit -m "feat: add minimal QA training chain"
git push
```
