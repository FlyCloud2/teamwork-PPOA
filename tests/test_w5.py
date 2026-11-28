"""Regression tests for the W5 K-sweep tooling.

Covers the sweep matrix (which cells and which command lines it produces), the
one-factor verification at four arms, and the aggregation ranking. The synthetic
runs are built by the same helper as the W4 tests and go through the same
``verify_run`` checks as real runs, so a test cannot pass on evidence the
production verifier would reject.
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src"), str(REPO / "tests")]
import run_kscan  # noqa: E402
import summarize_kscan  # noqa: E402
from test_w4 import ENVS, SEEDS, write_run  # noqa: E402
from verify_runs import KSCAN_EPOCHS, verify_kscan  # noqa: E402

ARMS = (1, 3, 10, 20)


def write_kscan(root, epochs=KSCAN_EPOCHS):
    """A complete synthetic sweep: every cell a full 1,000,000-step run."""
    for value in epochs:
        for env in ENVS:
            for seed in SEEDS:
                base = 10.0 + 2.0 * value
                write_run(root, env, seed, label=f"k{value}", epochs=value,
                          returns=(base, base + 5.0, base + 10.0),
                          steps=1_000_000, num_steps=2048, updates=489)


def run_ids(root):
    return sorted(path.name for path in (Path(root) / "runs").iterdir())


class MatrixTests(unittest.TestCase):
    def test_matrix_covers_every_cell_exactly_once(self):
        cells = run_kscan.matrix(Path("output"), ARMS, 1)
        self.assertEqual(len(cells), len(ENVS) * len(SEEDS) * len(ARMS))
        keys = [cell for cell, _ in cells]
        self.assertEqual(len(set(keys)), len(keys))
        self.assertEqual(set(keys),
                         {(env, seed, value) for env in ENVS for seed in SEEDS
                          for value in ARMS})

    def test_every_command_sets_the_epoch_count_and_its_label(self):
        for (env, seed, value), command in run_kscan.matrix(Path("output"), ARMS, 1):
            self.assertEqual(command[command.index("--update-epochs") + 1], str(value))
            self.assertEqual(command[command.index("--label") + 1], f"k{value}")
            self.assertEqual(command[command.index("--env") + 1], env)
            self.assertEqual(command[command.index("--seed") + 1], str(seed))
            self.assertEqual(command[command.index("--total-timesteps") + 1], "1000000")
            self.assertEqual(command[command.index("--variant") + 1], "clip0.2")
            self.assertEqual(command[command.index("--cuda") + 1], "false")
            self.assertNotIn("--smoke", command)

    def test_epochs_are_sorted_and_de_duplicated(self):
        self.assertEqual(run_kscan.parse_epochs([10, 1, 3, 1]), (1, 3, 10))
        with self.assertRaises(SystemExit):
            run_kscan.parse_epochs([0])

    def test_a_job_count_above_the_cell_count_is_refused(self):
        with self.assertRaises(SystemExit):
            run_kscan.main(["--jobs", str(len(ENVS) * len(SEEDS) * len(ARMS) + 1),
                            "--dry-run", "--output-root", "unused"])

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = run_kscan.main(["--dry-run", "--output-root", root])
            self.assertEqual(code, 0)
            self.assertFalse((Path(root) / "runs").exists())
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["cells"], len(ENVS) * len(SEEDS) * len(ARMS))
            self.assertEqual(len(payload["commands"]), len(ENVS) * len(SEEDS) * len(ARMS))


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        write_kscan(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_complete_sweep_passes(self):
        configs = verify_kscan(self.root, run_ids(self.root), ARMS)
        self.assertEqual(len(configs), len(ENVS) * len(SEEDS) * len(ARMS))

    def test_a_missing_cell_is_rejected(self):
        ids = run_ids(self.root)[1:]
        with self.assertRaisesRegex(ValueError, "36 distinct runs"):
            verify_kscan(self.root, ids, ARMS)

    def test_a_second_factor_is_rejected(self):
        victim = self.root / "runs" / run_ids(self.root)[0] / "config.json"
        config = json.loads(victim.read_text(encoding="utf-8"))
        config["hyperparameters"]["learning_rate"] = 1e-3
        victim.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            verify_kscan(self.root, run_ids(self.root), ARMS)

    def test_a_two_arm_subset_is_not_a_sweep(self):
        with self.assertRaisesRegex(ValueError, "distinct runs"):
            verify_kscan(self.root, run_ids(self.root), (1, 3))


class SummaryTests(unittest.TestCase):
    def test_ranking_is_best_to_worst_by_mean(self):
        per_env = {"Hopper-v4": {1: {"mean": 10.0}, 3: {"mean": 30.0}, 10: {"mean": 20.0}}}
        self.assertEqual(summarize_kscan.rank_arms(per_env)["Hopper-v4"], [3, 10, 1])

    def test_incomplete_summary_is_marked_incomplete(self):
        per_env = {"Hopper-v4": {value: {"mean": float(value)} for value in ARMS}}
        summary = summarize_kscan.build_summary(per_env, [], ARMS)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["best_counts"]["k20"], 1)

    def test_problems_never_produce_a_complete_summary(self):
        per_env = {env: {value: {"mean": 1.0} for value in ARMS} for env in ENVS}
        summary = summarize_kscan.build_summary(per_env, ["Hopper-v4/k3/seed1: missing"], ARMS)
        self.assertFalse(summary["complete"])


if __name__ == "__main__":
    unittest.main()
