import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "system" / "find_logs_by_best_acc.py"
SPEC = importlib.util.spec_from_file_location("find_logs_by_best_acc", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_log(path, algorithm="FedCLIP", model="Decom_CNN-5-512", best="0.55213"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "=" * 50,
                f"algorithm = {algorithm}",
                f"model_family = {model}",
                "=" * 50,
                "Best accuracy.",
                best,
                "All done!",
            ]
        ),
        encoding="utf-8",
    )


class FindLogsByBestAccuracyTest(unittest.TestCase):
    def test_parse_log_uses_last_best_accuracy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "20260101_010101_01.log"
            write_log(path, best="0.50")
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\nBest accuracy.\n0.56123\n")

            parsed = MODULE.parse_log(path)

            self.assertIsNotNone(parsed)
            self.assertEqual(parsed.algorithm, "FedCLIP")
            self.assertEqual(parsed.model, "Decom_CNN-5-512")
            self.assertEqual(parsed.best_accuracy_text, "0.56123")

    def test_percentage_and_decimal_prefixes_are_equivalent(self):
        self.assertEqual(MODULE.normalize_accuracy_prefix("55.21"), "0.5521")
        self.assertEqual(MODULE.normalize_accuracy_prefix("55.21%"), "0.5521")
        self.assertEqual(MODULE.normalize_accuracy_prefix("0.5521"), "0.5521")
        self.assertEqual(MODULE.normalize_accuracy_prefix("0.550"), "0.550")

    def test_high_precision_percentage_prefix_is_preserved(self):
        prefix = MODULE.normalize_accuracy_prefix("55.21092")

        self.assertEqual(prefix, "0.5521092")
        self.assertTrue("0.552109234567".startswith(prefix))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_log(root / "20260101_010101_01.log", best="0.552109234567")

            matches, _, _ = MODULE.find_matching_logs(
                root, "FedCLIP", "Decom_CNN-5-512", ["55.21092"]
            )

            self.assertEqual(len(matches), 1)

    def test_filters_algorithm_and_model_and_returns_every_prefix_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_log(root / "20260101_010101_01.log", best="0.55213")
            write_log(root / "20260102_010101_01.log", best="0.55219")
            write_log(root / "20260103_010101_01.log", algorithm="FedAvg", best="0.55218")
            write_log(root / "20260104_010101_01.log", model="CNN-5-512", best="0.55217")

            matches, scanned, parsed = MODULE.find_matching_logs(
                root, "FedCLIP", "Decom_CNN-5-512", ["55.21"]
            )

            self.assertEqual(scanned, 4)
            self.assertEqual(parsed, 4)
            self.assertEqual(len(matches), 2)

    def test_matches_are_sorted_by_experiment_time_newest_first(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            older = root / "Cifar100" / "FedCLIP" / "cfg" / "20260101_010101_01.log"
            newer = root / "Cifar100" / "FedCLIP" / "cfg" / "20260201_010101_01.log"
            write_log(older)
            write_log(newer)

            matches, _, _ = MODULE.find_matching_logs(
                root, "FedCLIP", "Decom_CNN-5-512", ["0.55"]
            )

            self.assertEqual([item[0].path for item in matches], [newer.resolve(), older.resolve()])

    def test_cli_prints_each_matching_absolute_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "20260101_010101_01.log"
            second = root / "20260102_010101_01.log"
            write_log(first)
            write_log(second)
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = MODULE.main(
                    [
                        "--algorithm",
                        "FedCLIP",
                        "--model",
                        "Decom_CNN-5-512",
                        "--best-acc",
                        "0.5521",
                        "--log-root",
                        str(root),
                    ]
                )

            text = output.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn(str(first.resolve()), text)
            self.assertIn(str(second.resolve()), text)
            self.assertLess(text.index(str(second.resolve())), text.index(str(first.resolve())))


if __name__ == "__main__":
    unittest.main()
