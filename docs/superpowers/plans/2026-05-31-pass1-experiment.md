# Pass 1 实验运行器实现计划

> **给后续执行者的要求：** 实现本计划时，使用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans`，按任务逐步执行。每个步骤用 checkbox 追踪。

**目标：** 新增一个正式实验入口，用来跑 DuReader_robust 的 Pass 1 模型筛选预实验；同时保留 `scripts/minimal_qa.py` 作为最小链路验证脚本。

**核心架构：** 不重写一套问答训练代码。已有的 `scripts/minimal_qa.py` 继续负责数据加载、特征构建、训练、预测和官方评测；新增 `scripts/run_experiment.py` 负责正式实验的模型组别、运行名称、输出目录和元数据记录。

**技术栈：** Python 3.9+、PyTorch、Transformers、DuReader_robust 官方 `evaluate.py`、`unittest`。

---

## 先解释：这个计划到底是什么

这个文件不是实验结果，也不是已经跑完了 Pass 1。

它只是下一阶段“怎么写代码”的实施清单。也就是说：

- 现在已经有的是：实验设计、最小 QA 链路、这个实现计划。
- 下一步要做的是：按这个计划写 `scripts/run_experiment.py`。
- 写完以后先跑一个很小的 smoke test，确认新脚本能工作。
- smoke test 通过后，再问你是否开始真正跑完整 Pass 1。
- 真正的 Pass 1 是 A/B/C 三个模型各跑一次 `train 1000 / dev 300 / epoch 1`。

Pass 1 仍然不是论文最终主实验。它是模型筛选预实验，用来决定 Pass 2 主实验该扩大哪个模型和训练规模。

## 本阶段范围

实现完成后，下面三条命令应该可以使用：

```bash
.venv/bin/python scripts/run_experiment.py --model-key A_L2_H128
.venv/bin/python scripts/run_experiment.py --model-key B_L4_H256
.venv/bin/python scripts/run_experiment.py --model-key C_L4_H512
```

默认配置是：

- 训练样本：`train_samples = 1000`
- 验证样本：`eval_samples = 300`
- 训练轮数：`epochs = 1`
- 批大小：`batch_size = 8`
- 最大长度：`max_length = 256`
- 滑动窗口步长：`doc_stride = 64`
- 学习率：`learning_rate = 3e-5`
- 随机种子：`seed = 42`
- 输出目录：`outputs/experiments`

生成的预测文件、模型输出、数据子集和指标 JSON 都放在 `outputs/` 下面，不提交到 Git。

## 文件分工

- 修改 `scripts/minimal_qa.py`
  - 保留原来的最小链路验证功能。
  - 增加模型参数量统计。
  - 增加分阶段耗时统计，例如数据加载、模型加载、特征构建、训练、预测、官方评测。

- 新增 `scripts/run_experiment.py`
  - 作为正式实验入口。
  - 把 `A_L2_H128`、`B_L4_H256`、`C_L4_H512` 映射到 HuggingFace 模型名。
  - 自动生成运行名称和输出目录。
  - 写入 `run_config.json`。
  - 调用 `minimal_qa.train_and_predict()` 复用已有训练链路。
  - 在 `metrics.json` 里补充实验阶段、模型组别、运行名称等信息。

- 修改 `tests/test_minimal_qa.py`
  - 增加一个模型参数量统计的纯函数测试。

- 新增 `tests/test_run_experiment.py`
  - 测试模型组别解析、运行名称生成、输出目录生成、训练参数构造。

- 修改 `README.md`
  - 增加 Pass 1 模型筛选命令。
  - 明确 Pass 1 不是最终主实验。

- 修改 `scripts/README.md`
  - 说明 `run_experiment.py` 是正式实验入口。

## 任务 1：给 `minimal_qa.py` 增加模型大小和分阶段耗时

**涉及文件：**

- 修改：`scripts/minimal_qa.py`
- 修改：`tests/test_minimal_qa.py`

### Step 1：先写失败测试

在 `tests/test_minimal_qa.py` 的 `MinimalQaTest` 类里添加：

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

运行：

```bash
.venv/bin/python -m unittest tests.test_minimal_qa.MinimalQaTest.test_model_parameter_stats_counts_total_trainable_and_bytes -v
```

预期结果：失败，报错原因是 `minimal_qa` 里还没有 `model_parameter_stats`。

### Step 2：实现模型参数量统计函数

在 `scripts/minimal_qa.py` 的 `choose_device()` 后面添加：

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

这个函数用于论文里的“模型大小”指标，不依赖训练结果。

### Step 3：给训练链路增加分阶段耗时

在 `train_and_predict()` 里增加这些计时字段：

- `data_seconds`：读取数据、切分 dev 子集、展平样本耗时。
- `model_load_seconds`：加载 tokenizer 和模型耗时。
- `feature_seconds`：构造训练/验证特征耗时。
- `training_seconds`：训练耗时。
- `prediction_seconds`：模型预测和答案抽取耗时。
- `evaluation_seconds`：调用官方评测脚本耗时。
- `seconds`：保留已有总耗时字段。

同时在 `metrics.json` 里加入：

```python
        "model_parameter_stats": model_stats,
        "data_seconds": data_seconds,
        "model_load_seconds": model_load_seconds,
        "feature_seconds": feature_seconds,
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "evaluation_seconds": evaluation_seconds,
```

### Step 4：验证任务 1

运行：

```bash
.venv/bin/python -m unittest tests.test_minimal_qa.MinimalQaTest.test_model_parameter_stats_counts_total_trainable_and_bytes -v
```

预期结果：通过。

## 任务 2：新增正式实验入口 `run_experiment.py`

**涉及文件：**

- 新增：`scripts/run_experiment.py`
- 新增：`tests/test_run_experiment.py`

### Step 1：先写失败测试

创建 `tests/test_run_experiment.py`：

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

运行：

```bash
.venv/bin/python -m unittest tests.test_run_experiment -v
```

预期结果：失败，因为 `scripts/run_experiment.py` 还不存在。

### Step 2：实现 `scripts/run_experiment.py`

创建 `scripts/run_experiment.py`：

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

### Step 3：验证任务 2

运行：

```bash
.venv/bin/python -m unittest tests.test_run_experiment -v
```

预期结果：通过。

## 任务 3：更新中文说明文档

**涉及文件：**

- 修改：`README.md`
- 修改：`scripts/README.md`

### Step 1：更新 `scripts/README.md`

脚本列表改成：

```markdown
- `inspect_dureader.py`：下载 DuReader_robust，统计数据划分，并验证官方 dev 集 F1/EM 评测脚本。
- `minimal_qa.py`：运行极小规模的模型训练、预测、评测链路，用于确认流程能跑通。
- `run_experiment.py`：正式实验入口，用模型组别、样本规模和运行名管理实验输出。
```

### Step 2：更新 `README.md`

在正式实验主线后加入：

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

## 任务 4：实现后如何验证

实现完成后要跑：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

预期结果：所有单元测试通过。

然后跑一个很小的 smoke test：

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

预期结果：

- 命令退出码为 0。
- 生成 `outputs/experiments/smoke_A_L2_H128/run_config.json`。
- 生成 `outputs/experiments/smoke_A_L2_H128/metrics.json`。
- `metrics.json` 里包含：
  - `stage: "smoke"`
  - `is_main_experiment: false`
  - `model_key: "A_L2_H128"`
  - `official_metrics`
  - `model_parameter_stats`
  - 各阶段耗时字段

如果模型下载或网络访问卡住，根据仓库规则先运行：

```bash
source /Users/luojiaqiang/script/proxy_on.sh
```

再重试 smoke test。

## 任务 5：实现后如何提交

只提交代码、测试和文档，不提交 `outputs/`、模型缓存、预测文件或原始数据。

提交前检查：

```bash
git diff --check
git status --short
```

确认 `outputs/experiments/smoke_A_L2_H128/` 没有出现在 Git 状态里。

提交命令：

```bash
git add scripts/minimal_qa.py scripts/run_experiment.py tests/test_minimal_qa.py tests/test_run_experiment.py README.md scripts/README.md
git commit -m "feat: add Pass 1 experiment runner"
git push
```

## 实现完成后还不能马上做什么

实现并通过 smoke test 后，还不能直接开始完整 Pass 1。

完整 Pass 1 要单独等用户确认，因为下面三条命令会比 smoke test 慢很多：

```bash
.venv/bin/python scripts/run_experiment.py --model-key A_L2_H128
.venv/bin/python scripts/run_experiment.py --model-key B_L4_H256
.venv/bin/python scripts/run_experiment.py --model-key C_L4_H512
```

三组 Pass 1 全部跑完后，再进入新的阶段：整理 `docs/results/experiment_summary.md`，把 A/B/C 的 F1、EM、耗时、模型大小汇总成论文可用的表格。
