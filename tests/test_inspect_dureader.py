import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "inspect_dureader.py"
spec = importlib.util.spec_from_file_location("inspect_dureader", SCRIPT_PATH)
inspect_dureader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspect_dureader)


class InspectDureaderTest(unittest.TestCase):
    def test_summarize_squad_like_dataset_counts_and_lengths(self):
        dataset = {
            "data": [
                {
                    "paragraphs": [
                        {
                            "context": "韩国全称大韩民国。",
                            "qas": [
                                {
                                    "question": "韩国全称",
                                    "id": "q1",
                                    "answers": [{"text": "大韩民国", "answer_start": 4}],
                                }
                            ],
                        },
                        {
                            "context": "北京是中国的首都。",
                            "qas": [
                                {
                                    "question": "中国首都",
                                    "id": "q2",
                                    "answers": [{"text": "北京", "answer_start": 0}],
                                },
                                {
                                    "question": "北京是什么",
                                    "id": "q3",
                                    "answers": [{"text": "首都", "answer_start": 6}],
                                },
                            ],
                        },
                    ]
                }
            ]
        }

        summary = inspect_dureader.summarize_squad_like_dataset("train", dataset)

        self.assertEqual(summary["split"], "train")
        self.assertEqual(summary["paragraphs"], 2)
        self.assertEqual(summary["qas"], 3)
        self.assertEqual(summary["answers"], 3)
        self.assertEqual(summary["avg_context_chars"], 9.0)
        self.assertEqual(summary["avg_question_chars"], 4.33)
        self.assertEqual(summary["avg_answer_chars"], 2.67)

    def test_build_perfect_predictions_uses_first_answer_text(self):
        dataset = {
            "data": [
                {
                    "paragraphs": [
                        {
                            "context": "韩国全称大韩民国。",
                            "qas": [
                                {
                                    "question": "韩国全称",
                                    "id": "q1",
                                    "answers": [{"text": "大韩民国", "answer_start": 4}],
                                }
                            ],
                        }
                    ]
                }
            ]
        }

        predictions = inspect_dureader.build_first_answer_predictions(dataset)

        self.assertEqual(predictions, {"q1": "大韩民国"})


if __name__ == "__main__":
    unittest.main()
