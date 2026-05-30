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


if __name__ == "__main__":
    unittest.main()
