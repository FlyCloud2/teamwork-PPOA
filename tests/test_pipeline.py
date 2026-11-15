"""Regression tests for configuration, exact budgets and evidence integrity."""
import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
import run
from ppo_args import parse_args
from verify_runs import verify_run


class ArgumentTests(unittest.TestCase):
    def test_shared_defaults_and_explicit_forwarding(self):
        args = run.parse_runner_args(["--env", "Hopper-v4", "--update-epochs", "3", "--gamma", "0.9",
                                      "--anneal-lr", "false", "--hidden-sizes", "32", "16", "--adam-eps", "0.0001"])
        cmd, hp = run.child_command(args, Path("output"))
        child = parse_args(cmd[3:])
        for key, value in hp.items():
            self.assertEqual(getattr(child, key), value, key)
        self.assertEqual(child.env_id, "Hopper-v4")
        self.assertFalse(child.anneal_lr)
        self.assertEqual(child.num_updates, 489)
        self.assertEqual(child.hidden_sizes, [32, 16])

    def test_invalid_parameters_fail_before_creating_artifacts(self):
        cases = [["--threads", "0"], ["--num-steps", "0"], ["--num-minibatches", "31"],
                 ["--num-envs", "3"], ["--total-timesteps", "1"], ["--gamma", "nan"],
                 ["--learning-rate", "inf"], ["--variant", "adaptiveKL"],
                 ["--clip-coef", "0.1"], ["--cuda", "garbage"]]
        for case in cases:
            with self.subTest(case=case), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run.parse_runner_args(case)

    def test_dry_run_creates_nothing(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run.main(["--dry-run", "--output-root", directory]), 0)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_smoke_exact_budget(self):
        args = run.parse_runner_args(["--smoke"])
        self.assertEqual(args.total_timesteps, 20_000)
        self.assertEqual(args.num_updates, 10)


class EvidenceTests(unittest.TestCase):
    def test_index_updates_without_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run.update_index({"run_id": "a", "status": "running"}, root)
            run.update_index({"run_id": "b", "status": "failed"}, root)
            run.update_index({"run_id": "a", "status": "completed"}, root)
            with (root / "logs/runs.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["status"], "completed")

    def test_concurrent_index_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            code = "import sys; from pathlib import Path; sys.path.insert(0, 'scripts'); from run import update_index; update_index({'run_id':sys.argv[2], 'status':'completed'}, Path(sys.argv[1]))"
            children = [subprocess.Popen([sys.executable, "-c", code, directory, str(i)], cwd=REPO) for i in range(8)]
            self.assertEqual([child.wait() for child in children], [0] * 8)
            with (Path(directory) / "logs/runs.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 8)

    def test_truncated_and_nonfinite_progress_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.csv"
            fields = ["global_step", "policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "learning_rate", "wall_time_s"]
            hp = {"batch_size": 64, "total_timesteps": 96}
            def write(steps, loss="1"):
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    for step in steps:
                        writer.writerow(dict.fromkeys(fields, loss) | {"global_step": step})
            write([64])
            with self.assertRaises(ValueError):
                run.inspect_progress(path, hp)
            write([64, 96], "nan")
            with self.assertRaises(ValueError):
                run.inspect_progress(path, hp)
            write([64, 96])
            self.assertEqual(run.inspect_progress(path, hp), {"actual_steps": 96, "updates": 2})

    def test_real_training_handles_partial_rollout_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, str(REPO / "scripts/run.py"), "--total-timesteps", "160",
                       "--num-steps", "64", "--num-minibatches", "4", "--update-epochs", "1",
                       "--threads", "1", "--output-root", directory]
            result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, errors="replace")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            path = next((Path(directory) / "runs").iterdir())
            config = verify_run(path, Path(directory))
            self.assertEqual(config["actual_steps"], 160)
            self.assertEqual(config["updates"], 3)
            self.assertEqual(config["hyperparameters"]["update_epochs"], 1)
            config["hyperparameters"]["batch_size"] = 65
            (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "batch size"):
                verify_run(path, Path(directory))

    def test_real_vector_training_with_partial_rollout(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(REPO / "scripts/run.py"), "--env", "Hopper-v4",
                                     "--num-envs", "2", "--total-timesteps", "160", "--num-steps", "32",
                                     "--num-minibatches", "4", "--update-epochs", "1", "--threads", "1",
                                     "--output-root", directory], cwd=REPO, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            path = next((Path(directory) / "runs").iterdir())
            config = verify_run(path, Path(directory))
            self.assertEqual(config["actual_steps"], 160)
            self.assertEqual(config["updates"], 3)

    def test_failed_child_is_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(REPO / "scripts/run.py"), "--env", "NoSuchEnvironment-v0",
                                     "--output-root", directory], cwd=REPO, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            path = next((Path(directory) / "runs").iterdir())
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["status"], "failed")
            with (Path(directory) / "logs/runs.csv").open() as handle:
                self.assertEqual(next(csv.DictReader(handle))["status"], "failed")


if __name__ == "__main__":
    unittest.main()
