"""Regression tests for the W4 evidence tooling.

Covers the run label that makes C2's two arms distinguishable, the C2 matrix
verification (one factor only), run selection, and the generated
Paper-vs-Ours table. The synthetic runs are fully consistent artifacts: they go
through the same ``verify_run`` checks as real runs, so a test cannot pass on
evidence the production verifier would reject.
"""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
import evidence  # noqa: E402
import make_table  # noqa: E402
import run  # noqa: E402
import summarize_c2  # noqa: E402
from verify_runs import verify_c2, verify_run  # noqa: E402

ENVS = ("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4")
SEEDS = (0, 1, 2)
TIERS = ("clip0.2", "adaptiveKL", "noclip")
STEPS, NUM_STEPS, UPDATES = 192, 64, 3
PROGRESS_FIELDS = ["global_step", "episode_return", "policy_loss", "value_loss", "entropy",
                   "approx_kl", "clipfrac", "explained_variance", "learning_rate", "wall_time_s"]


def write_run(root, env, seed, variant="clip0.2", label="", epochs=10, returns=(1.0, 2.0, 3.0),
              learning_rate=3e-4, date="20261121-120000", steps=STEPS, num_steps=NUM_STEPS,
              updates=UPDATES):
    """A minimal but fully consistent run directory: config, CSV, log and index row."""
    rid = (f"{date}-000000_{env}_seed{seed}_{variant}"
           + (f"_{label}" if label else "") + "_full")
    path = Path(root) / "runs" / rid
    path.mkdir(parents=True)
    config = {
        "schema_version": 2, "run_id": rid, "status": "completed", "return_code": 0,
        "env": env, "seed": seed, "steps": steps, "variant": variant, "label": label,
        "smoke": False, "threads": 1, "actual_steps": steps, "updates": updates,
        "wall_time_s": 12.5, "git_commit_short": "abc1234",
        "hyperparameters": {
            "env_id": env, "seed": seed, "total_timesteps": steps, "num_envs": 1,
            "num_steps": num_steps, "num_minibatches": 32, "update_epochs": epochs,
            "learning_rate": learning_rate, "batch_size": num_steps,
            "minibatch_size": num_steps // 32, "num_updates": updates,
        },
        "packages": {"torch": "2.4.1", "gymnasium": "0.29.1", "mujoco": "3.1.6"},
        "host": {"platform": "test", "python": "3.10", "cpu_count": 1},
        "started": "2026-11-21T12:00:00",
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    # Global steps mirror the real exact-budget sequence: full batches, then a
    # shorter final batch that lands exactly on total_timesteps.
    boundaries = [min(num_steps * (index + 1), steps) for index in range(updates)]
    with (path / "progress.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROGRESS_FIELDS)
        writer.writeheader()
        for index, boundary in enumerate(boundaries):
            writer.writerow({"global_step": boundary,
                             "episode_return": returns[index % len(returns)],
                             "policy_loss": "-0.1", "value_loss": "1.0", "entropy": "0.5",
                             "approx_kl": "0.01", "clipfrac": "0.1",
                             "explained_variance": "0.2", "learning_rate": "3e-4",
                             "wall_time_s": "1.0"})
    (path / "stdout.log").write_text("SPS: 100\n", encoding="utf-8")
    run.update_index(config, Path(root))
    return path


def write_c1(root):
    for tier, returns in (("clip0.2", (30.0, 20.0, 10.0)),
                          ("adaptiveKL", (7.0, 6.0, 5.0)),
                          ("noclip", (3.0, 2.0, 1.0))):
        for env in ENVS:
            for seed in SEEDS:
                write_run(root, env, seed, variant=tier, returns=returns)


def write_c2(root):
    """The real C2 budget: 1,000,000 steps = 488 full batches + one 576-step batch."""
    for epochs in (1, 10):
        for env in ENVS:
            for seed in SEEDS:
                base = 90.0 if epochs == 10 else 10.0
                write_run(root, env, seed, variant="clip0.2", label=f"k{epochs}",
                          epochs=epochs, returns=(base, base + 5.0, base + 10.0),
                          steps=1_000_000, num_steps=2048, updates=489)


class LabelTests(unittest.TestCase):
    def test_label_enters_run_id_but_not_the_trainer(self):
        args = run.parse_runner_args(["--label", "k1"])
        rid = run.run_id("Hopper-v4", 2, "clip0.2", args.label)
        self.assertTrue(rid.endswith("_Hopper-v4_seed2_clip0.2_k1_full"), rid)
        self.assertFalse(run.run_id("Hopper-v4", 2, "clip0.2").endswith("_k1_full"))
        command, _ = run.child_command(args, Path("output"))
        self.assertNotIn("--label", command)

    def test_unsafe_labels_are_rejected_before_any_directory_exists(self):
        for label in ("a b", "k/1", "k\\1", "-k1", "k" * 25, "中文"):
            with self.subTest(label=label), self.assertRaises(SystemExit):
                run.parse_runner_args(["--label", label])

    def test_label_is_recorded_in_a_real_run(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(REPO / "scripts/run.py"), "--total-timesteps", "160",
                 "--num-steps", "64", "--num-minibatches", "4", "--update-epochs", "1",
                 "--label", "k1", "--threads", "1", "--output-root", directory],
                cwd=REPO, capture_output=True, text=True, errors="replace")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            path = next((Path(directory) / "runs").iterdir())
            self.assertTrue(path.name.endswith("_clip0.2_k1_full"), path.name)
            config = verify_run(path, Path(directory))
            self.assertEqual(config["label"], "k1")


class SelectionTests(unittest.TestCase):
    def test_find_run_refuses_a_duplicated_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            write_run(directory, "Hopper-v4", 0, label="k1", epochs=1)
            self.assertTrue(evidence.find_run(directory, "Hopper-v4", 0, "clip0.2", "k1"))
            write_run(directory, "Hopper-v4", 0, label="k1", epochs=1,
                      date="20261121-130000")
            with self.assertRaises(evidence.EvidenceError):
                evidence.find_run(directory, "Hopper-v4", 0, "clip0.2", "k1")

    def test_partial_matrix_is_reported_as_a_problem_not_a_row(self):
        with tempfile.TemporaryDirectory() as directory:
            write_run(directory, "Hopper-v4", 0, label="k1", epochs=1)
            write_run(directory, "Hopper-v4", 1, label="k1", epochs=1)
            rows, problems = evidence.collect(directory, envs=("Hopper-v4",), variant="clip0.2",
                                              label="k1", validator=verify_run)
            self.assertEqual(rows, [])
            self.assertEqual(len(problems), 1)
            self.assertIn("seed2", problems[0])

    def test_tail_metric_ignores_nan_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_run(directory, "Hopper-v4", 0)
            rows = list(csv.DictReader((path / "progress.csv").open(encoding="utf-8", newline="")))
            rows[1]["episode_return"] = "nan"
            with (path / "progress.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=PROGRESS_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(evidence.tail_metric(path / "progress.csv"), 2.0)


class C2VerificationTests(unittest.TestCase):
    def ids(self, root):
        return sorted(path.name for path in (Path(root) / "runs").iterdir())

    def test_complete_matrix_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            write_c2(directory)
            configs = verify_c2(Path(directory), self.ids(directory))
            self.assertEqual(len(configs), 18)
            self.assertEqual({config["hyperparameters"]["update_epochs"] for config in configs},
                             {1, 10})

    def test_a_second_factor_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            write_c2(directory)
            ids = self.ids(directory)
            victim = Path(directory) / "runs" / ids[0] / "config.json"
            config = json.loads(victim.read_text(encoding="utf-8"))
            config["hyperparameters"]["learning_rate"] = 1e-4
            victim.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "update_epochs"):
                verify_c2(Path(directory), ids)

    def test_incomplete_matrix_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            write_c2(directory)
            ids = self.ids(directory)
            with self.assertRaises(ValueError):
                verify_c2(Path(directory), ids[1:])

    def test_inconsistent_arm_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            write_c2(directory)
            ids = self.ids(directory)
            victim = Path(directory) / "runs" / ids[0] / "config.json"
            config = json.loads(victim.read_text(encoding="utf-8"))
            current = config["hyperparameters"]["update_epochs"]
            config["hyperparameters"]["update_epochs"] = 10 if current == 1 else 1
            victim.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_c2(Path(directory), ids)


class TableTests(unittest.TestCase):
    def test_table_and_json_are_generated_from_the_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            write_c1(directory)
            write_c2(directory)
            out = Path(directory) / "docs" / "table.md"
            payload = Path(directory) / "logs" / "table.json"
            self.assertEqual(make_table.main(["--root", directory, "--out", str(out),
                                              "--json", str(payload)]), 0)
            text = out.read_text(encoding="utf-8")
            for heading in ("Table 1.", "Table 2.", "Table 3.", "Table 4a.", "Table 4b.",
                            "Table 5."):
                self.assertIn(heading, text)
            self.assertIn("| 1.00 |", text)                 # clip is the best tier everywhere
            self.assertIn("K = 10 wins in 3/3 tasks", text)
            data = json.loads(payload.read_text(encoding="utf-8"))
            self.assertEqual(data["c2"]["k10_wins"], 3)
            self.assertEqual(data["c1"]["normalised_mean"]["clip0.2"], 1.0)
            self.assertEqual(len(data["c1"]["rows"]), 3)

    def test_partial_matrix_writes_no_table(self):
        with tempfile.TemporaryDirectory() as directory:
            write_c1(directory)
            write_c2(directory)
            next((Path(directory) / "runs").glob("*_noclip_full")).rename(
                Path(directory) / "runs" / "quarantined")
            out = Path(directory) / "docs" / "table.md"
            self.assertEqual(make_table.main(["--root", directory, "--out", str(out)]), 1)
            self.assertFalse(out.exists())


class SummaryTests(unittest.TestCase):
    def test_c2_summary_reports_both_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            write_c2(directory)
            out = Path(directory) / "logs" / "c2.json"
            self.assertEqual(summarize_c2.main(["--root", directory, "--out", str(out)]), 0)
            summary = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(summary["problems"], [])
            self.assertEqual(summary["k10_wins"], 3)
            self.assertEqual([row["env"] for row in summary["rows"]], list(ENVS))
            self.assertGreater(summary["rows"][0]["delta"], 0)


if __name__ == "__main__":
    unittest.main()
