import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "system" / "find_log_by_model.py"
SPEC = importlib.util.spec_from_file_location("find_log_by_model", SCRIPT)
finder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = finder
SPEC.loader.exec_module(finder)


class FindLogByModelTests(unittest.TestCase):
    def test_original_organized_and_archived_logs(self):
        ids = ["1787318160604562916_749305_0", "1787318160595649060_749312_0",
               "1787318160600654528_749319_0", "1787318160600493358_749326_0"]
        queries = [finder.Query.parse(f"./final_models/runs/{run}/Client_0_model.pt")
                   for run in ids]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / "20260821_211553_01_FedCLIP_Cifar100_pat_cpc20" / "train.log"
            original.parent.mkdir()
            original.write_text(f"Saved ./final_models/runs/{ids[0]}/\n", encoding="utf-8")
            organized = root / "log" / "Cifar100" / "FedCLIP" / "pat_cpc20"
            organized.mkdir(parents=True)
            (organized / "20260821_211553_02.log").write_text(
                f"Saved final_models\\runs\\{ids[1]}\\Client_0_model.pt", encoding="utf-8")
            with zipfile.ZipFile(root / "8-21-03.zip", "w") as archive:
                for index in (2, 3):
                    archive.writestr(f"20260821_211553_0{index+1}/train.log",
                                     f"run_id: {ids[index]}\n")
            results, count, errors = finder.find_logs(queries, [root], include_zips=True)
            self.assertEqual(count, 4)
            self.assertFalse(errors)
            self.assertTrue(all(len(matches) == 1 for matches in results))
            self.assertTrue(all(matches[0].kind == "run_id" for matches in results))
            self.assertIn("::", results[2][0].path)

    def test_pid_reuse_is_not_an_exact_match(self):
        query = finder.Query.parse("1787318160604562916_749305_0")
        lines = ["other run 9999999999999999999_749305_0\n",
                 "wrong trial 1787318160604562916_749305_01\n"]
        exact, fallback = finder.scan_lines(lines, "train.log", 0, [query], True)
        self.assertIsNone(exact[0])
        self.assertEqual(fallback[0].kind, "pid_only")

    def test_newest_experiment_first_despite_mtime(self):
        query = finder.Query.parse("1787318160604562916_749305_0")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("20260822_100000.log", "20260821_100000.log"):
                (root / name).write_text(query.run_id, encoding="utf-8")
            results, _, _ = finder.find_logs([query], [root, root])
            self.assertEqual(len(results[0]), 2)
            self.assertTrue(results[0][0].path.endswith("20260822_100000.log"))

    def test_exact_match_suppresses_weak_pid_candidates(self):
        query = finder.Query.parse("1787318160604562916_749305_0")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "exact.log").write_text(query.run_id, encoding="utf-8")
            (root / "reused.log").write_text("PID: 749305", encoding="utf-8")
            results, _, _ = finder.find_logs([query], [root], allow_pid_fallback=True)
            self.assertEqual(len(results[0]), 1)
            self.assertEqual(results[0][0].kind, "run_id")


if __name__ == "__main__":
    unittest.main()
