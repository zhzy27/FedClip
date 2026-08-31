import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def find_bash():
    bash = shutil.which("bash")
    if bash:
        return bash
    git = shutil.which("git")
    if os.name == "nt" and git:
        candidate = Path(git).resolve().parents[1] / "bin" / "bash.exe"
        if candidate.is_file():
            return str(candidate)
    return None


@unittest.skipUnless(find_bash(), "Bash is required to test the launcher")
class TsneLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tsne launcher ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.system = self.root / "system"
        self.system.mkdir()
        for filename in ("run_tsne.sh", "tsne_params.example.sh"):
            shutil.copyfile(ROOT / "system" / filename, self.system / filename)
        self.config = self.system / "tsne_params.local.sh"
        self.env = os.environ.copy()
        example = (self.system / "tsne_params.example.sh").read_text(encoding="utf-8")
        for name in re.findall(r"^([A-Z_]+)=", example, re.MULTILINE):
            self.env.pop(name, None)
        self.env["PYTHON_BIN"] = Path(sys.executable).as_posix()
        # Exercise exec/argument forwarding without loading models or plotting.
        (self.system / "T-SNE-Cifar-legacy-compatible.py").write_text(
            'import json, os, sys\n'
            'print("PAYLOAD=" + json.dumps({"cwd": os.getcwd(), "args": sys.argv[1:]}))\n',
            encoding="utf-8",
        )

    def launch(self, *args, env=None, cwd=None):
        return subprocess.run(
            [find_bash(), (self.system / "run_tsne.sh").as_posix(), *args],
            cwd=cwd or self.root, env=env or self.env,
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )

    def payload(self, result):
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        line = next(line for line in result.stdout.splitlines() if line.startswith("PAYLOAD="))
        payload = json.loads(line.removeprefix("PAYLOAD="))
        self.assertEqual(Path(payload["cwd"]).resolve(), self.system.resolve())
        return payload["args"]

    def test_default_config_and_missing_settings_use_example_defaults(self):
        self.config.write_text('PERPLEXITY=15\n', encoding="utf-8")
        args = self.payload(self.launch())
        self.assertEqual(args[args.index("--perplexity") + 1], "15")
        self.assertEqual(args[args.index("--dataset") + 1], "Cifar100")
        self.assertEqual(args[args.index("--model-family") + 1], "Decom_resnet18_5")
        self.assertIn("--show-legend", args)
        self.assertIn("--no-save-excel", args)

    def test_dataset_class_counts_and_flags(self):
        for dataset, count in (("Cifar10", "10"), ("Cifar100", "100"), ("TinyImagenet", "200")):
            with self.subTest(dataset=dataset):
                self.config.write_text(
                    f'DATASET={dataset}\nAUTO_BEST_CLIENT=1\nSHOW_LEGEND=0\nSAVE_EXCEL=1\n',
                    encoding="utf-8",
                )
                args = self.payload(self.launch())
                self.assertEqual(args[args.index("--num-classes") + 1], count)
                self.assertIn("--auto-best-client", args)
                self.assertIn("--no-show-legend", args)
                self.assertIn("--save-excel", args)

    def test_custom_config_path_spaces_and_empty_client_selection(self):
        custom = self.root / "another experiment.local.sh"
        custom.write_text(
            'CLIENT_IDS=""\nMODEL_FAMILY=""\nMODEL_DIR="./models/run A"\n'
            'OUTPUT_DIR="./plots/experiment A"\nSPLIT=train\n', encoding="utf-8",
        )
        args = self.payload(self.launch("--config", custom.name))
        for key, expected in (("--client-ids", ""), ("--model-family", ""),
                              ("--model-dir", "./models/run A"),
                              ("--output-dir", "./plots/experiment A"), ("--split", "train")):
            self.assertEqual(args[args.index(key) + 1], expected)

    def test_environment_override_and_default_config_from_other_cwd(self):
        shutil.copyfile(self.system / "tsne_params.example.sh", self.config)
        env = dict(self.env, DATASET="Cifar10", CLIENT_IDS="", PERPLEXITY="50")
        args = self.payload(self.launch(env=env, cwd=self.system))
        self.assertEqual(args[args.index("--num-classes") + 1], "10")
        self.assertEqual(args[args.index("--client-ids") + 1], "")
        self.assertEqual(args[args.index("--perplexity") + 1], "50")

    def test_dry_run_accepts_either_option_order_without_executing(self):
        self.config.write_text("SPLIT=train\n", encoding="utf-8")
        for args in (("--dry-run", "--config", self.config.as_posix()),
                     ("--config", self.config.as_posix(), "--dry-run")):
            result = self.launch(*args)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--split train", result.stdout)
            self.assertNotIn("PAYLOAD=", result.stdout)

    def test_best_client_is_forwarded_without_numeric_conversion(self):
        self.config.write_text('CLIENT_IDS="best"\nSPLIT=train\n', encoding="utf-8")
        args = self.payload(self.launch())
        self.assertEqual(args[args.index("--client-ids") + 1], "best")
        self.assertEqual(args[args.index("--split") + 1], "train")

    def test_missing_config_fails_and_help_does_not_require_config(self):
        result = self.launch("--dry-run")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Settings file not found", result.stderr)
        self.assertNotIn("PAYLOAD=", result.stdout)
        result = self.launch("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("Git-ignored", result.stdout)

    def test_invalid_arguments_and_config_values_fail_before_exec(self):
        self.config.write_text("", encoding="utf-8")
        for args in (("--unknown",), ("--config",), ("--config", "--dry-run")):
            self.assertEqual(self.launch(*args).returncode, 2)
        for config in ("DATASET=wrong\n", "SHOW_LEGEND=2\n"):
            self.config.write_text(config, encoding="utf-8")
            result = self.launch()
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("PAYLOAD=", result.stdout)


if __name__ == "__main__":
    unittest.main()
