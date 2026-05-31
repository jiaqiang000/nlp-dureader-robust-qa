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
