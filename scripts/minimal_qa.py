#!/usr/bin/env python3
"""Run a tiny DuReader_robust extractive QA training chain."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_MODEL = "uer/chinese_roberta_L-2_H-128"
DEFAULT_DATASET_DIR = Path("data/raw/dureader_robust-data")
DEFAULT_OUTPUT_DIR = Path("outputs/minimal_qa")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def iter_paragraphs(dataset: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for article in dataset.get("data", []):
        for paragraph in article.get("paragraphs", []):
            yield paragraph


def flatten_examples(dataset: Dict[str, Any], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for paragraph in iter_paragraphs(dataset):
        context = paragraph.get("context", "")
        for qa in paragraph.get("qas", []):
            examples.append(
                {
                    "id": str(qa["id"]),
                    "question": qa.get("question", ""),
                    "context": context,
                    "answers": qa.get("answers", []),
                }
            )
            if limit is not None and len(examples) >= limit:
                return examples
    return examples


def subset_dataset(dataset: Dict[str, Any], limit: int) -> Dict[str, Any]:
    remaining = limit
    paragraphs: List[Dict[str, Any]] = []
    for paragraph in iter_paragraphs(dataset):
        if remaining <= 0:
            break
        qas = paragraph.get("qas", [])
        selected_qas = qas[:remaining]
        if selected_qas:
            copied = dict(paragraph)
            copied["qas"] = selected_qas
            paragraphs.append(copied)
            remaining -= len(selected_qas)
    return {"data": [{"title": "", "paragraphs": paragraphs}]}


def locate_answer_token_span(
    offsets: Sequence[Tuple[int, int]],
    sequence_ids: Sequence[Optional[int]],
    answer_start: int,
    answer_text: str,
    cls_index: int,
) -> Tuple[int, int]:
    if not answer_text:
        return cls_index, cls_index

    answer_end = answer_start + len(answer_text)
    context_indices = [idx for idx, sequence_id in enumerate(sequence_ids) if sequence_id == 1]
    if not context_indices:
        return cls_index, cls_index

    context_start = context_indices[0]
    context_end = context_indices[-1]
    if offsets[context_start][0] > answer_start or offsets[context_end][1] < answer_end:
        return cls_index, cls_index

    start_token = None
    for idx in context_indices:
        start, end = offsets[idx]
        if start <= answer_start < end:
            start_token = idx
            break

    end_token = None
    for idx in reversed(context_indices):
        start, end = offsets[idx]
        if start < answer_end <= end:
            end_token = idx
            break

    if start_token is None or end_token is None:
        return cls_index, cls_index
    return start_token, end_token


def pick_best_answer(
    context: str,
    features: Sequence[Dict[str, Any]],
    n_best_size: int = 20,
    max_answer_length: int = 30,
) -> str:
    best_score = float("-inf")
    best_answer = ""

    for feature in features:
        offsets = feature["offset_mapping"]
        start_logits = feature["start_logits"]
        end_logits = feature["end_logits"]
        start_indexes = sorted(range(len(start_logits)), key=lambda idx: start_logits[idx], reverse=True)[:n_best_size]
        end_indexes = sorted(range(len(end_logits)), key=lambda idx: end_logits[idx], reverse=True)[:n_best_size]

        for start_index in start_indexes:
            for end_index in end_indexes:
                if end_index < start_index:
                    continue
                if end_index - start_index + 1 > max_answer_length:
                    continue
                if start_index >= len(offsets) or end_index >= len(offsets):
                    continue
                char_start = offsets[start_index][0]
                char_end = offsets[end_index][1]
                if char_end <= char_start:
                    continue
                score = float(start_logits[start_index]) + float(end_logits[end_index])
                if score > best_score:
                    best_score = score
                    best_answer = context[char_start:char_end]

    return best_answer


def _normalize_offsets(offsets: Sequence[Tuple[int, int]], sequence_ids: Sequence[Optional[int]]) -> List[Tuple[int, int]]:
    return [tuple(offset) if sequence_id == 1 else (0, 0) for offset, sequence_id in zip(offsets, sequence_ids)]


def prepare_train_features(
    examples: Sequence[Dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    doc_stride: int,
) -> List[Dict[str, Any]]:
    features: List[Dict[str, Any]] = []
    for example in examples:
        answers = example.get("answers", [])
        if not answers:
            continue
        answer = answers[0]
        encoded = tokenizer(
            example["question"],
            example["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        for feature_index, input_ids in enumerate(encoded["input_ids"]):
            offsets = encoded["offset_mapping"][feature_index]
            sequence_ids = encoded.sequence_ids(feature_index)
            cls_index = input_ids.index(tokenizer.cls_token_id)
            start_position, end_position = locate_answer_token_span(
                offsets=offsets,
                sequence_ids=sequence_ids,
                answer_start=answer["answer_start"],
                answer_text=answer["text"],
                cls_index=cls_index,
            )
            feature = {
                "input_ids": input_ids,
                "attention_mask": encoded["attention_mask"][feature_index],
                "start_positions": start_position,
                "end_positions": end_position,
            }
            if "token_type_ids" in encoded:
                feature["token_type_ids"] = encoded["token_type_ids"][feature_index]
            features.append(feature)
    return features


def prepare_eval_features(
    examples: Sequence[Dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    doc_stride: int,
) -> List[Dict[str, Any]]:
    features: List[Dict[str, Any]] = []
    for example in examples:
        encoded = tokenizer(
            example["question"],
            example["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        for feature_index, input_ids in enumerate(encoded["input_ids"]):
            sequence_ids = encoded.sequence_ids(feature_index)
            feature = {
                "example_id": example["id"],
                "input_ids": input_ids,
                "attention_mask": encoded["attention_mask"][feature_index],
                "offset_mapping": _normalize_offsets(encoded["offset_mapping"][feature_index], sequence_ids),
            }
            if "token_type_ids" in encoded:
                feature["token_type_ids"] = encoded["token_type_ids"][feature_index]
            features.append(feature)
    return features


def evaluate_official(dataset_path: Path, predictions_path: Path, evaluate_path: Path) -> Dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(evaluate_path), str(dataset_path), str(predictions_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def choose_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def train_and_predict(args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    class FeatureDataset(Dataset):
        def __init__(self, features: Sequence[Dict[str, Any]], include_labels: bool):
            self.features = features
            self.include_labels = include_labels

        def __len__(self) -> int:
            return len(self.features)

        def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
            feature = self.features[index]
            keys = ["input_ids", "attention_mask"]
            if "token_type_ids" in feature:
                keys.append("token_type_ids")
            item = {key: torch.tensor(feature[key], dtype=torch.long) for key in keys}
            if self.include_labels:
                item["start_positions"] = torch.tensor(feature["start_positions"], dtype=torch.long)
                item["end_positions"] = torch.tensor(feature["end_positions"], dtype=torch.long)
            return item

    random.seed(args.seed)
    torch.manual_seed(args.seed)
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

    device = choose_device()
    model.to(device)
    model.train()
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        FeatureDataset(train_features, include_labels=True),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

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

    evaluation_start = time.time()
    metrics = evaluate_official(dev_subset_path, predictions_path, args.dataset_dir / "evaluate.py")
    evaluation_seconds = round(time.time() - evaluation_start, 2)
    result = {
        "model_name": args.model_name,
        "device": device,
        "train_examples": len(train_examples),
        "train_features": len(train_features),
        "eval_examples": len(dev_examples),
        "eval_features": len(eval_features),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "doc_stride": args.doc_stride,
        "seed": args.seed,
        "last_train_loss": last_loss,
        "model_parameter_stats": model_stats,
        "data_seconds": data_seconds,
        "model_load_seconds": model_load_seconds,
        "feature_seconds": feature_seconds,
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "evaluation_seconds": evaluation_seconds,
        "seconds": round(time.time() - start_time, 2),
        "official_metrics": metrics,
    }
    save_json(output_dir / "metrics.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--train-samples", type=int, default=80)
    parser.add_argument("--eval-samples", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--doc-stride", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_and_predict(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
