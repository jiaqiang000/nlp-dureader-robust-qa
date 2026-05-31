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
