#!/usr/bin/env python3
"""Download and inspect the DuReader_robust dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_DATA_URL = "https://bj.bcebos.com/paddlenlp/datasets/dureader_robust-data.tar.gz"
DEFAULT_DATA_DIR = Path("data/raw")
ARCHIVE_NAME = "dureader_robust-data.tar.gz"
EXTRACTED_NAME = "dureader_robust-data"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        with destination.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)


def _safe_tar_members(tar: tarfile.TarFile, target_dir: Path) -> Iterable[tarfile.TarInfo]:
    target_root = target_dir.resolve()
    for member in tar.getmembers():
        member_path = (target_dir / member.name).resolve()
        if target_root not in [member_path, *member_path.parents]:
            raise ValueError(f"Unsafe tar member path: {member.name}")
        yield member


def extract_archive(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(target_dir, members=_safe_tar_members(tar, target_dir))


def ensure_dataset(data_dir: Path, url: str, force_download: bool = False) -> Path:
    archive_path = data_dir / ARCHIVE_NAME
    extracted_dir = data_dir / EXTRACTED_NAME

    if force_download or not archive_path.exists():
        download_file(url, archive_path)

    if force_download or not extracted_dir.exists():
        extract_archive(archive_path, data_dir)

    return extracted_dir


def iter_paragraphs(dataset: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for article in dataset.get("data", []):
        for paragraph in article.get("paragraphs", []):
            yield paragraph


def _average(values: List[int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def summarize_squad_like_dataset(split: str, dataset: Dict[str, Any]) -> Dict[str, Any]:
    paragraphs = list(iter_paragraphs(dataset))
    qas = [qa for paragraph in paragraphs for qa in paragraph.get("qas", [])]
    answers = [
        answer
        for qa in qas
        for answer in qa.get("answers", [])
        if isinstance(answer, dict) and "text" in answer
    ]

    context_lengths = [len(paragraph.get("context", "")) for paragraph in paragraphs]
    question_lengths = [len(qa.get("question", "")) for qa in qas]
    answer_lengths = [len(answer.get("text", "")) for answer in answers]

    return {
        "split": split,
        "paragraphs": len(paragraphs),
        "qas": len(qas),
        "answers": len(answers),
        "avg_context_chars": _average(context_lengths),
        "avg_question_chars": _average(question_lengths),
        "avg_answer_chars": _average(answer_lengths),
        "max_context_chars": max(context_lengths, default=0),
        "max_question_chars": max(question_lengths, default=0),
        "max_answer_chars": max(answer_lengths, default=0),
    }


def build_first_answer_predictions(dataset: Dict[str, Any]) -> Dict[str, str]:
    predictions = {}
    for paragraph in iter_paragraphs(dataset):
        for qa in paragraph.get("qas", []):
            answers = qa.get("answers", [])
            if answers and isinstance(answers[0], dict):
                predictions[str(qa["id"])] = answers[0].get("text", "")
    return predictions


def run_official_evaluate(dataset_path: Path, predictions_path: Path, evaluate_path: Path) -> Dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(evaluate_path), str(dataset_path), str(predictions_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def inspect_dataset(data_dir: Path, url: str, force_download: bool = False) -> Dict[str, Any]:
    dataset_dir = ensure_dataset(data_dir, url, force_download=force_download)

    split_files = {
        "train": dataset_dir / "train.json",
        "dev": dataset_dir / "dev.json",
        "test": dataset_dir / "test.json",
    }
    summaries = {
        split: summarize_squad_like_dataset(split, load_json(path))
        for split, path in split_files.items()
        if path.exists()
    }

    dev_path = split_files["dev"]
    dev_dataset = load_json(dev_path)
    predictions_path = data_dir / "dev_first_answer_predictions.json"
    save_json(predictions_path, build_first_answer_predictions(dev_dataset))

    evaluation = run_official_evaluate(
        dev_path,
        predictions_path,
        dataset_dir / "evaluate.py",
    )

    return {
        "dataset_dir": str(dataset_dir),
        "summaries": summaries,
        "dev_first_answer_evaluation": evaluation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--url", default=DEFAULT_DATA_URL)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = inspect_dataset(args.data_dir, args.url, force_download=args.force_download)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
