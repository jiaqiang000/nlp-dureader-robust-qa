# Pass 1 Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a formal Pass 1 model-screening experiment runner for DuReader_robust while keeping `scripts/minimal_qa.py` as the minimal chain verification entry.

**Architecture:** `scripts/minimal_qa.py` remains the shared QA training/evaluation implementation. A new `scripts/run_experiment.py` wraps it with experiment naming, model-group selection, output-directory rules, and Pass 1 metadata. Tests cover pure helper behavior; a tiny smoke run verifies the full chain without treating smoke metrics as paper results.

**Tech Stack:** Python 3.9+, PyTorch, Transformers, DuReader_robust official `evaluate.py`, `unittest`.

---

## Scope

Pass 1 is a model-screening pre-experiment, not the paper's final main experiment.

The implementation must make these commands possible after user confirmation:

```bash
.venv/bin/python scripts/run_experiment.py --model-key A_L2_H128
.venv/bin/python scripts/run_experiment.py --model-key B_L4_H256
.venv/bin/python scripts/run_experiment.py --model-key C_L4_H512
```

By default each command uses:

- `train_samples = 1000`
- `eval_samples = 300`
- `epochs = 1`
- `batch_size = 8`
- `max_length = 256`
- `doc_stride = 64`
- `learning_rate = 3e-5`
- `seed = 42`
- `output_root = outputs/experiments`

Generated model outputs, predictions, subsets, and metrics stay under `outputs/` and do not enter Git.

## File Structure

- Modify `scripts/minimal_qa.py`
  - Keep the existing CLI behavior for tiny chain verification.
  - Add model parameter statistics.
  - Add phase-level timing fields to `metrics.json`.
- Create `scripts/run_experiment.py`
  - Formal experiment entrypoint.
  - Maps short model keys to HuggingFace model names.
  - Writes `run_config.json`.
  - Calls `minimal_qa.train_and_predict`.
  - Rewrites `metrics.json` with experiment metadata.
- Modify `tests/test_minimal_qa.py`
  - Add a pure unit test for parameter statistics.
- Create `tests/test_run_experiment.py`
  - Test model-key resolution, run-name construction, output directory construction, and training-args construction.
- Modify `README.md`
  - Add Pass 1 command examples and clarify that Pass 1 is screening, not the final main experiment.
- Modify `scripts/README.md`
  - Document the new formal experiment script.

## Task 1: Add Model Size And Phase Timing To `minimal_qa.py`

**Files:**

- Modify: `scripts/minimal_qa.py`
- Modify: `tests/test_minimal_qa.py`

- [ ] **Step 1: Write the failing parameter-statistics test**

Add this test method to `tests/test_minimal_qa.py` inside `MinimalQaTest`:

```python
    def test_model_parameter_stats_counts_total_trainable_and_bytes(self):
        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(3, 2, bias=False),
            torch.nn.Linear(2, 1, bias=True),
        )
        model[1].bias.requires_grad = False

        stats = minimal_qa.model_parameter_stats(model)

        self.assertEqual(stats["total_parameters"], 9)
        self.assertEqual(stats["trainable_parameters"], 8)
        self.assertEqual(stats["parameter_bytes"], 36)
        self.assertEqual(stats["parameter_size_mb"], 0.0)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
.venv/bin/python -m unittest tests.test_minimal_qa.MinimalQaTest.test_model_parameter_stats_counts_total_trainable_and_bytes -v
```

Expected: FAIL with `AttributeError: module 'minimal_qa' has no attribute 'model_parameter_stats'`.

- [ ] **Step 3: Add the minimal parameter-statistics implementation**

Add this function to `scripts/minimal_qa.py` after `choose_device()`:

```python
def model_parameter_stats(model: Any) -> Dict[str, Any]:
    parameters = list(model.parameters())
    total_parameters = sum(parameter.numel() for parameter in parameters)
    trainable_parameters = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    return {
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "parameter_bytes": int(parameter_bytes),
        "parameter_size_mb": round(parameter_bytes / 1024 / 1024, 2),
    }
```

- [ ] **Step 4: Add phase timing and parameter stats to `train_and_predict`**

In `scripts/minimal_qa.py`, update the body of `train_and_predict` with these timing boundaries:

```python
    start_time = time.time()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_start = time.time()
    train_dataset = load_json(args.dataset_dir / "train.json")
    dev_dataset = load_json(args.dataset_dir / "dev.json")
    dev_subset = subset_dataset(dev_dataset, args.eval_samples)
    dev_subset_path = output_dir / "dev_subset.json"
    save_json(dev_subset_path, dev_subset)

    train_examples = flatten_examples(train_dataset, args.train_samples)
    dev_examples = flatten_examples(dev_subset, args.eval_samples)
    contexts_by_id = {example["id"]: example["context"] for example in dev_examples}
    data_seconds = round(time.time() - data_start, 2)

    model_load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
    model_stats = model_parameter_stats(model)
    model_load_seconds = round(time.time() - model_load_start, 2)

    feature_start = time.time()
    train_features = prepare_train_features(train_examples, tokenizer, args.max_length, args.doc_stride)
    eval_features = prepare_eval_features(dev_examples, tokenizer, args.max_length, args.doc_stride)
    feature_seconds = round(time.time() - feature_start, 2)
```

Wrap the training loop with:

```python
    training_start = time.time()
    last_loss = None
    for _epoch in range(args.epochs):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            last_loss = float(loss.detach().cpu())
    training_seconds = round(time.time() - training_start, 2)
```

Wrap prediction and answer extraction with:

```python
    prediction_start = time.time()
    model.eval()
    eval_loader = DataLoader(
        FeatureDataset(eval_features, include_labels=False),
        batch_size=args.batch_size,
        shuffle=False,
    )
    predicted_features: List[Dict[str, Any]] = []
    feature_offset = 0
    with torch.no_grad():
        for batch in eval_loader:
            batch_on_device = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch_on_device)
            start_logits = outputs.start_logits.detach().cpu().tolist()
            end_logits = outputs.end_logits.detach().cpu().tolist()
            for row_index in range(len(start_logits)):
                feature = dict(eval_features[feature_offset + row_index])
                feature["start_logits"] = start_logits[row_index]
                feature["end_logits"] = end_logits[row_index]
                predicted_features.append(feature)
            feature_offset += len(start_logits)

    features_by_example: Dict[str, List[Dict[str, Any]]] = {}
    for feature in predicted_features:
        features_by_example.setdefault(feature["example_id"], []).append(feature)

    predictions = {
        example_id: pick_best_answer(contexts_by_id[example_id], features)
        for example_id, features in features_by_example.items()
    }
    predictions_path = output_dir / "predictions.json"
    save_json(predictions_path, predictions)
    prediction_seconds = round(time.time() - prediction_start, 2)
```

Wrap official evaluation with:

```python
    evaluation_start = time.time()
    metrics = evaluate_official(dev_subset_path, predictions_path, args.dataset_dir / "evaluate.py")
    evaluation_seconds = round(time.time() - evaluation_start, 2)
```

Add these fields to the `result` dictionary:

```python
        "model_parameter_stats": model_stats,
        "data_seconds": data_seconds,
        "model_load_seconds": model_load_seconds,
        "feature_seconds": feature_seconds,
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "evaluation_seconds": evaluation_seconds,
```

Keep the existing `"seconds": round(time.time() - start_time, 2)` field for backward compatibility.

- [ ] **Step 5: Run the focused test and verify it passes**

Run:

```bash
.venv/bin/python -m unittest tests.test_minimal_qa.MinimalQaTest.test_model_parameter_stats_counts_total_trainable_and_bytes -v
```

Expected: PASS.

## Task 2: Add Formal Experiment Runner

**Files:**

- Create: `scripts/run_experiment.py`
- Create: `tests/test_run_experiment.py`

- [ ] **Step 1: Write failing tests for the experiment runner helpers**

Create `tests/test_run_experiment.py`:

```python
import argparse
import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py"
spec = importlib.util.spec_from_file_location("run_experiment", SCRIPT_PATH)
run_experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_experiment)


class RunExperimentTest(unittest.TestCase):
    def test_resolve_model_name_uses_group_mapping(self):
        model_name = run_experiment.resolve_model_name("A_L2_H128", None)

        self.assertEqual(model_name, "uer/chinese_roberta_L-2_H-128")

    def test_resolve_model_name_allows_explicit_override(self):
        model_name = run_experiment.resolve_model_name("A_L2_H128", "custom/model")

        self.assertEqual(model_name, "custom/model")

    def test_build_run_name_records_model_and_scale(self):
        run_name = run_experiment.build_run_name(
            model_key="B_L4_H256",
            train_samples=1000,
            eval_samples=300,
            epochs=1,
        )

        self.assertEqual(run_name, "B_L4_H256_train1000_dev300_e1")

    def test_build_output_dir_uses_output_root_and_run_name(self):
        output_dir = run_experiment.build_output_dir(Path("outputs/experiments"), "A_L2_H128_train1000_dev300_e1")

        self.assertEqual(output_dir, Path("outputs/experiments/A_L2_H128_train1000_dev300_e1"))

    def test_build_training_args_sets_formal_defaults(self):
        cli_args = argparse.Namespace(
            dataset_dir=Path("data/raw/dureader_robust-data"),
            train_samples=1000,
            eval_samples=300,
            epochs=1,
            batch_size=8,
            max_length=256,
            doc_stride=64,
            learning_rate=3e-5,
            seed=42,
        )

        training_args = run_experiment.build_training_args(
            cli_args=cli_args,
            output_dir=Path("outputs/experiments/A_L2_H128_train1000_dev300_e1"),
            model_name="uer/chinese_roberta_L-2_H-128",
        )

        self.assertEqual(training_args.dataset_dir, Path("data/raw/dureader_robust-data"))
        self.assertEqual(training_args.output_dir, Path("outputs/experiments/A_L2_H128_train1000_dev300_e1"))
        self.assertEqual(training_args.model_name, "uer/chinese_roberta_L-2_H-128")
        self.assertEqual(training_args.train_samples, 1000)
        self.assertEqual(training_args.eval_samples, 300)
        self.assertEqual(training_args.epochs, 1)
        self.assertEqual(training_args.batch_size, 8)
        self.assertEqual(training_args.max_length, 256)
        self.assertEqual(training_args.doc_stride, 64)
        self.assertEqual(training_args.learning_rate, 3e-5)
        self.assertEqual(training_args.seed, 42)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
.venv/bin/python -m unittest tests.test_run_experiment -v
```

Expected: FAIL because `scripts/run_experiment.py` does not exist.

- [ ] **Step 3: Create `scripts/run_experiment.py`**

Create `scripts/run_experiment.py`:

```python
#!/usr/bin/env python3
"""Run formal DuReader_robust QA experiment configurations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from minimal_qa import DEFAULT_DATASET_DIR, save_json, train_and_predict


MODEL_GROUPS: Dict[str, str] = {
    "A_L2_H128": "uer/chinese_roberta_L-2_H-128",
    "B_L4_H256": "uer/chinese_roberta_L-4_H-256",
    "C_L4_H512": "uer/chinese_roberta_L-4_H-512",
}

DEFAULT_OUTPUT_ROOT = Path("outputs/experiments")
DEFAULT_STAGE = "pass1_model_screening"


def resolve_model_name(model_key: str, model_name: Optional[str]) -> str:
    if model_name:
        return model_name
    return MODEL_GROUPS[model_key]


def build_run_name(model_key: str, train_samples: int, eval_samples: int, epochs: int) -> str:
    return f"{model_key}_train{train_samples}_dev{eval_samples}_e{epochs}"


def build_output_dir(output_root: Path, run_name: str) -> Path:
    return output_root / run_name


def build_training_args(cli_args: argparse.Namespace, output_dir: Path, model_name: str) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_dir=cli_args.dataset_dir,
        output_dir=output_dir,
        model_name=model_name,
        train_samples=cli_args.train_samples,
        eval_samples=cli_args.eval_samples,
        epochs=cli_args.epochs,
        batch_size=cli_args.batch_size,
        max_length=cli_args.max_length,
        doc_stride=cli_args.doc_stride,
        learning_rate=cli_args.learning_rate,
        seed=cli_args.seed,
    )


def build_run_config(
    args: argparse.Namespace,
    run_name: str,
    output_dir: Path,
    model_name: str,
) -> Dict[str, Any]:
    return {
        "stage": args.stage,
        "is_main_experiment": args.stage == "main_experiment",
        "run_name": run_name,
        "model_key": args.model_key,
        "model_name": model_name,
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(output_dir),
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "doc_stride": args.doc_stride,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }


def run_single_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    model_name = resolve_model_name(args.model_key, args.model_name)
    run_name = args.run_name or build_run_name(args.model_key, args.train_samples, args.eval_samples, args.epochs)
    output_dir = build_output_dir(args.output_root, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = build_run_config(args, run_name, output_dir, model_name)
    save_json(output_dir / "run_config.json", run_config)

    training_args = build_training_args(args, output_dir, model_name)
    metrics = train_and_predict(training_args)
    metrics.update(run_config)
    save_json(output_dir / "metrics.json", metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-key", choices=sorted(MODEL_GROUPS), required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--stage", choices=["pass1_model_screening", "main_experiment", "smoke"], default=DEFAULT_STAGE)
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--doc-stride", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_single_experiment(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the new tests and verify they pass**

Run:

```bash
.venv/bin/python -m unittest tests.test_run_experiment -v
```

Expected: PASS.

## Task 3: Document The Formal Runner

**Files:**

- Modify: `README.md`
- Modify: `scripts/README.md`

- [ ] **Step 1: Update `scripts/README.md`**

Change the current script list to include:

```markdown
- `inspect_dureader.py`：下载 DuReader_robust，统计数据划分，并验证官方 dev 集 F1/EM 评测脚本。
- `minimal_qa.py`：运行极小规模的模型训练、预测、评测链路，用于确认流程能跑通。
- `run_experiment.py`：正式实验入口，用模型组别、样本规模和运行名管理实验输出。
```

- [ ] **Step 2: Update `README.md` with Pass 1 commands**

Add this section after the current formal experiment mainline:

````markdown
## Pass 1 模型筛选命令

Pass 1 是模型筛选预实验，不是最终主实验。默认配置为 `train 1000 / dev 300 / epoch 1`。

```bash
python scripts/run_experiment.py --model-key A_L2_H128
python scripts/run_experiment.py --model-key B_L4_H256
python scripts/run_experiment.py --model-key C_L4_H512
```

每次运行会写入：

```text
outputs/experiments/<run_name>/
├── run_config.json
├── metrics.json
├── predictions.json
└── dev_subset.json
```

这些输出文件不进入 Git。完成 Pass 1 后，再把汇总指标写入 `docs/results/experiment_summary.md`。
````

- [ ] **Step 3: Check Markdown formatting**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

## Task 4: Verify The Implementation

**Files:**

- No new files.

- [ ] **Step 1: Run all unit tests**

Run:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 2: Run a tiny smoke experiment**

Run:

```bash
.venv/bin/python scripts/run_experiment.py \
  --stage smoke \
  --model-key A_L2_H128 \
  --train-samples 4 \
  --eval-samples 2 \
  --epochs 1 \
  --batch-size 2 \
  --output-root outputs/experiments \
  --run-name smoke_A_L2_H128
```

Expected:

- Command exits with code 0.
- `outputs/experiments/smoke_A_L2_H128/run_config.json` exists.
- `outputs/experiments/smoke_A_L2_H128/metrics.json` exists.
- `metrics.json` includes `stage: "smoke"`, `is_main_experiment: false`, `model_key: "A_L2_H128"`, `official_metrics`, `model_parameter_stats`, and phase timing fields.

If model download or network access stalls, follow the repository instruction and run:

```bash
source /Users/luojiaqiang/script/proxy_on.sh
```

Then retry the smoke command.

- [ ] **Step 3: Confirm ignored outputs are not staged**

Run:

```bash
git status --short
```

Expected: code/docs files may be modified before commit; `outputs/experiments/smoke_A_L2_H128/` must not appear because `outputs/` is ignored.

## Task 5: Commit The Implementation

**Files:**

- Stage only code, tests, and documentation.

- [ ] **Step 1: Review changed files**

Run:

```bash
git diff --stat
git diff -- scripts/minimal_qa.py scripts/run_experiment.py tests/test_minimal_qa.py tests/test_run_experiment.py README.md scripts/README.md
```

Expected: changes are limited to the formal experiment runner, metric metadata, tests, and docs.

- [ ] **Step 2: Commit**

Run:

```bash
git add scripts/minimal_qa.py scripts/run_experiment.py tests/test_minimal_qa.py tests/test_run_experiment.py README.md scripts/README.md
git commit -m "feat: add Pass 1 experiment runner"
```

Expected: commit succeeds.

- [ ] **Step 3: Push**

Run:

```bash
git push
```

Expected: push succeeds.

## After Implementation: Run Pass 1 Only After User Confirmation

After the implementation commit is pushed, ask before starting the real Pass 1 runs. The real Pass 1 commands are:

```bash
.venv/bin/python scripts/run_experiment.py --model-key A_L2_H128
.venv/bin/python scripts/run_experiment.py --model-key B_L4_H256
.venv/bin/python scripts/run_experiment.py --model-key C_L4_H512
```

These runs may take noticeably longer than the smoke test. They generate experiment outputs but should not commit raw predictions, model caches, or `outputs/` contents.

After all three Pass 1 runs complete, create a separate results-summary task for `docs/results/experiment_summary.md`.
