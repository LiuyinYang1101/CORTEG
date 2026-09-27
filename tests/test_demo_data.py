"""Checks on the interactive demo's data (docs/data) and on build_demo_data.py.

`TestShippedDemoData` reads the committed JSON and checks what docs/index.html
relies on: every manifest model is present for every subject, traces have the
target's length, the time axis matches the stated duration, and no value is
NaN or infinite. `TestBuilder` runs the builder on a few synthetic runs in a
temporary directory: a missing or inconsistent run fails the build without
touching the files already there, rows on another grid and rows whose paper
value is a seed mean are checked like the others, and the builder replaces
only directories it wrote itself. Everything runs on CPU in a few seconds and
needs none of the run outputs the demo is built from.

Run:  python -m unittest tests.test_demo_data -v
"""
from __future__ import annotations

import io
import json
import math
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DOCS_DATA = REPO / "docs" / "data"
sys.path.insert(0, str(REPO))

import build_demo_data as bdd  # noqa: E402


def _reject_constant(name):
    raise ValueError(f"JSON holds {name}, which the browser cannot parse")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def check_dataset(tc: unittest.TestCase, ds_dir: Path, fingers=None):
    """What docs/index.html needs from one dataset directory; returns the manifest."""
    man = load_json(ds_dir / "manifest.json")
    tc.assertGreater(len(man["models"]), 0, f"{ds_dir.name}: manifest lists no model")
    tc.assertEqual(sorted(p.stem for p in ds_dir.glob("*.json") if p.stem != "manifest"),
                   sorted(man["subjects"]), f"{ds_dir.name}: subject files vs manifest")
    for key in ("notes", "paper_r", "cohort_mean_r"):
        tc.assertEqual(list(man[key]), man["models"], f"{ds_dir.name}: manifest {key}")
    if fingers is not None:
        tc.assertEqual(man["fingers"], fingers)
    means = {m: [] for m in man["models"]}
    for sub in man["subjects"]:
        d = load_json(ds_dir / f"{sub}.json")
        where = f"{ds_dir.name}/{sub}"
        tc.assertEqual(list(d["models"]), man["models"], f"{where}: models vs manifest")
        n = len(d["y_true"])
        tc.assertGreater(n, 1, where)
        tc.assertGreater(d["fs_hz"], 0, where)
        tc.assertAlmostEqual((n - 1) / d["fs_hz"], d["duration_s"], delta=0.1,
                             msg=f"{where}: time axis vs duration")
        for label, e in d["models"].items():
            tc.assertEqual(len(e["y_pred"]), n, f"{where}/{label}: y_pred length")
            if fingers is not None:
                tc.assertEqual({len(row) for row in e["y_pred"]}, {len(fingers)}, where)
                r = e["corr_per_finger"]
                tc.assertEqual(len(r), len(fingers), where)
                tc.assertAlmostEqual(e["corr_mean"], sum(r) / len(r), delta=2e-6,
                                     msg=f"{where}/{label}: corr_mean")
                means[label].append(e["corr_mean"])
            else:
                r = [e["corr"]]
                means[label].append(e["corr"])
            tc.assertTrue(all(-1 <= v <= 1 for v in r), f"{where}/{label}: r out of range")
        if fingers is not None:
            tc.assertEqual({len(row) for row in d["y_true"]}, {len(fingers)}, where)
    for label, rs in means.items():
        tc.assertAlmostEqual(man["cohort_mean_r"][label], sum(rs) / len(rs), delta=2e-6,
                             msg=f"{ds_dir.name}/{label}: cohort_mean_r")
    return man


class TestShippedDemoData(unittest.TestCase):
    def test_each_dataset_is_complete(self):
        for name, cfg in bdd.DATASETS.items():
            with self.subTest(dataset=name):
                check_dataset(self, DOCS_DATA / name, cfg["fingers"])

    def test_manifest_matches_the_builder_config(self):
        """The shipped rows are the configured ones, with the configured paper values."""
        for name, cfg in bdd.DATASETS.items():
            man = load_json(DOCS_DATA / name / "manifest.json")
            self.assertEqual(man["subjects"], cfg["subjects"], name)
            self.assertEqual(man["models"], [m["label"] for m in cfg["models"]], name)
            self.assertEqual(man["paper_r"], {m["label"]: m["paper"] for m in cfg["models"]})

    def test_paper_rows_round_to_the_paper(self):
        """Each paper row rounds to the paper the way its config says, "exact" by default."""
        for name, cfg in bdd.DATASETS.items():
            man = load_json(DOCS_DATA / name / "manifest.json")
            for m in cfg["models"]:
                if m["paper"] is None or m.get("seed_mean_of"):
                    continue        # a seed mean needs the other seeds' runs
                with self.subTest(dataset=name, row=m["label"]):
                    self.assertEqual(
                        bdd.paper_rounding(man["cohort_mean_r"][m["label"]], m["paper"]),
                        m.get("paper_rounding", "exact"))

    def test_non_paper_rows_say_so(self):
        for name, cfg in bdd.DATASETS.items():
            for m in cfg["models"]:
                if m["paper"] is None:
                    self.assertIn("re-run", m["label"], f"{name}/{m['label']}")
                    self.assertTrue(m["note"].startswith("Not a paper value"), m["label"])

    def test_badges_tell_the_rows_apart(self):
        """index.html's prediction badge shows a label up to its first "(".

        That part alone must tell the rows apart and keep the disclosures a
        visitor relies on: a re-run says so, and a row that shows one seed of a
        paper's seed mean names the seed.
        """
        for name, cfg in bdd.DATASETS.items():
            badges = [m["label"].split("(")[0].strip() for m in cfg["models"]]
            self.assertEqual(len(set(badges)), len(badges), f"{name}: {badges}")
            for m, badge in zip(cfg["models"], badges):
                if m["paper"] is None:
                    self.assertIn("re-run", badge, f"{name}/{m['label']}")
                if m.get("seed_mean_of"):
                    self.assertIn("seed", badge, f"{name}/{m['label']}")


class TestPaperRounding(unittest.TestCase):
    def test_exact_and_intermediate_rounding(self):
        self.assertEqual(bdd.paper_rounding(0.5394418855716181, 0.539), "exact")
        self.assertEqual(bdd.paper_rounding(0.5534866677786353, 0.553), "exact")
        # The paper's 0.554 is reached only from 0.5535.
        self.assertEqual(bdd.paper_rounding(0.5534866677786353, 0.554), "via 4 dp")
        self.assertEqual(bdd.paper_rounding(0.24945, 0.250), "via 4 dp")
        self.assertIsNone(bdd.paper_rounding(0.5116, 0.539))

    def test_only_the_marked_row_may_round_via_4_dp(self):
        rows = [(name, m) for name, cfg in bdd.DATASETS.items() for m in cfg["models"]
                if "paper_rounding" in m]
        self.assertEqual([(name, m["label"], m["paper"], m["paper_rounding"])
                          for name, m in rows],
                         [("stanford", "CORTEG pooled ⭐", 0.554, "via 4 dp")])


class TestDisplayGrid(unittest.TestCase):
    def test_other_grid_is_sampled_by_time(self):
        """Ghent: CORTEG steps 6/128 s (3819 windows), the HiLoFuseNet re-run 50 ms (3580)."""
        n_ref, step_ref, n, step = 3819, 6 / 128, 3580, 0.05
        idx_ref = bdd.display_index(n_ref, step_ref, n_ref, step_ref)
        idx = bdd.display_index(n_ref, step_ref, n, step)
        self.assertEqual(len(idx), bdd.N_DISPLAY)
        self.assertLessEqual(np.abs(idx_ref * step_ref - idx * step).max(), step / 2 + 1e-9)
        self.assertEqual(idx.max(), n - 1)




def smooth_targets(rng, n, n_ch=2, width=10):
    """White noise through a `width`-point moving average: lag-1 autocorrelation
    about 1 - 1/width, like the smooth finger-flexion targets."""
    x = rng.standard_normal((n + width - 1, n_ch))
    return np.stack([np.convolve(x[:, i], np.ones(width) / width, mode="valid")
                     for i in range(n_ch)], axis=1)


class TestBuilder(unittest.TestCase):
    """build_demo_data.build_all on synthetic runs: two subjects, two channels."""

    SUBJECTS = ["s1", "s2"]
    N = 80
    STEP = 0.05

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="corteg_demo_test_"))
        self.root, self.out = self.tmp / "runs", self.tmp / "out"
        self.rng = np.random.default_rng(0)
        self.targets = {s: smooth_targets(self.rng, self.N) for s in self.SUBJECTS}
        a = self.make_run("run_a", noise=0.5)
        self.make_run("run_b", noise=1.0)
        self.mean_a = float(np.mean([np.mean(r) for r in a.values()]))
        self.paper_a = float(bdd.round_half_up(self.mean_a, 3))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_run(self, run, noise, every=1):
        """A run whose saved windows are every `every`-th window of the targets."""
        per_subject = {}
        for s in self.SUBJECTS:
            yt = self.targets[s][::every]
            per_subject[s] = self.save(run, s, yt, yt + noise * self.rng.standard_normal(yt.shape))
        return per_subject

    def save(self, run, sub, yt, yp):
        """Write one subject's predictions and store their r in the run's results
        file, with the run's score (the mean over subjects), as the runs do."""
        d = self.root / run / "predictions" / sub
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "y_true.npy", yt)
        np.save(d / "y_pred.npy", yp)
        path = self.root / run / "results.json"
        res = json.loads(path.read_text()) if path.exists() else {"per_subject": {}}
        r = bdd.pearson_per_channel(yt, yp)
        res["per_subject"][sub] = {"corr": r}
        res["score"] = float(np.mean([np.mean(v["corr"]) for v in res["per_subject"].values()]))
        path.write_text(json.dumps(res))
        return r

    def config(self, paper_a=None, **row_a):
        a = {"label": "A", "run": "run_a", "note": "paper row",
             "paper": self.paper_a if paper_a is None else paper_a, **row_a}
        return {"toy": {
            "subjects": self.SUBJECTS, "step_s": self.STEP, "fingers": ["a", "b"],
            "models": [a, {"label": "B (our re-run)", "run": "run_b", "paper": None,
                           "note": "Not a paper value."}]}}

    def build(self, cfg=None, **kw):
        """Run the builder; returns what it printed."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                bdd.build_all(cfg or self.config(), self.root, str(self.tmp), self.out, **kw)
            finally:
                self.printed = buf.getvalue()
        return self.printed

    def snapshot(self):
        if not self.out.exists():
            return None
        return {p.relative_to(self.out): p.read_bytes() for p in self.out.rglob("*") if p.is_file()}

    def assert_build_fails_and_writes_nothing(self, cfg=None, **kw):
        before = self.snapshot()
        names = sorted(p.name for p in self.out.iterdir()) if self.out.is_dir() else None
        with self.assertRaises(SystemExit) as cm:
            self.build(cfg, **kw)
        self.assertNotIn(cm.exception.code, (0, None))
        self.assertEqual(self.snapshot(), before, "a failed build changed the files")
        if names is not None:
            self.assertEqual(sorted(p.name for p in self.out.iterdir()), names,
                             "leftover directories")
        return str(cm.exception.code)

    def test_complete_runs_build(self):
        self.build()
        man = check_dataset(self, self.out / "toy", ["a", "b"])
        self.assertEqual(man["models"], ["A", "B (our re-run)"])
        self.build()                                    # a rebuild replaces in place
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["toy"])

    def test_missing_run_fails(self):
        self.build()
        shutil.rmtree(self.root / "run_b")
        self.assert_build_fails_and_writes_nothing()

    def test_missing_subject_fails(self):
        self.build()
        shutil.rmtree(self.root / "run_a" / "predictions" / "s2")
        self.assert_build_fails_and_writes_nothing()

    def test_empty_root_is_refused(self):
        self.build()
        self.root = self.tmp / "nowhere"
        self.assert_build_fails_and_writes_nothing()

    def test_wrong_paper_value_fails(self):
        self.build()
        self.assert_build_fails_and_writes_nothing(self.config(round(self.paper_a + 0.01, 3)))

    def test_paper_row_needs_its_stored_r(self):
        self.build()
        (self.root / "run_a" / "results.json").unlink()
        self.assert_build_fails_and_writes_nothing()

    def test_ambiguous_results_file_fails(self):
        self.build()
        shutil.copy(self.root / "run_a" / "results.json", self.root / "run_a" / "results_2.json")
        self.assert_build_fails_and_writes_nothing()

    def test_permuted_channels_fail(self):
        self.build()
        path = self.root / "run_a" / "results.json"
        res = json.loads(path.read_text())
        for ps in res["per_subject"].values():
            ps["corr"] = ps["corr"][::-1]
        path.write_text(json.dumps(res))
        self.assert_build_fails_and_writes_nothing()

    def test_misaligned_target_fails(self):
        self.build()
        # Both reversed, so run_b's own r is unchanged and only the alignment fails.
        d = self.root / "run_b" / "predictions" / "s1"
        self.save("run_b", "s1", np.load(d / "y_true.npy")[::-1], np.load(d / "y_pred.npy")[::-1])
        self.assert_build_fails_and_writes_nothing()

    def test_one_window_shift_on_the_same_grid_fails(self):
        """Same window count, target shifted by one window: the smooth target still
        correlates above the other-grid threshold, so only the same-grid one catches it."""
        self.build()
        d = self.root / "run_b" / "predictions" / "s1"
        yt, yp = (np.roll(np.load(d / f), 1, axis=0) for f in ("y_true.npy", "y_pred.npy"))
        ref = self.targets["s1"]
        shifted_r = min(np.corrcoef(ref[:, i], yt[:, i])[0, 1] for i in range(2))
        self.assertGreater(shifted_r, bdd.ALIGN_MIN_CORR_OTHER_GRID)
        self.save("run_b", "s1", yt, yp)
        msg = self.assert_build_fails_and_writes_nothing()
        self.assertIn("does not line up", msg)

    def test_non_paper_row_may_recompute_r(self):
        (self.root / "run_b" / "results.json").unlink()
        self.build()
        check_dataset(self, self.out / "toy", ["a", "b"])

    def test_allow_missing_drops_only_absent_rows(self):
        shutil.rmtree(self.root / "run_b")
        self.build(allow_missing=True)
        man = check_dataset(self, self.out / "toy", ["a", "b"])
        self.assertEqual(man["models"], ["A"])

    def test_allow_missing_may_drop_the_first_row(self):
        """The next row then sets the time axis."""
        shutil.rmtree(self.root / "run_a")
        self.build(allow_missing=True)
        man = check_dataset(self, self.out / "toy", ["a", "b"])
        self.assertEqual(man["models"], ["B (our re-run)"])

    # A row saved on another grid, like the Ghent HiLoFuseNet re-run.

    def other_grid_config(self):
        shutil.rmtree(self.root / "run_b")
        self.make_run("run_b", noise=1.0, every=2)          # 40 windows, 0.1 s apart
        cfg = self.config()
        cfg["toy"]["models"][1]["step_s"] = 2 * self.STEP
        return cfg

    def test_row_on_another_grid_builds(self):
        cfg = self.other_grid_config()
        self.build(cfg)
        man = check_dataset(self, self.out / "toy", ["a", "b"])
        self.assertEqual(man["models"], ["A", "B (our re-run)"])
        d = load_json(self.out / "toy" / "s1.json")
        self.assertEqual(len(d["models"]["B (our re-run)"]["y_pred"]), self.N)

    def test_row_on_another_grid_must_span_the_same_time(self):
        cfg = self.other_grid_config()
        self.build(cfg)
        d = self.root / "run_b" / "predictions" / "s1"
        self.save("run_b", "s1", np.load(d / "y_true.npy")[:-2], np.load(d / "y_pred.npy")[:-2])
        msg = self.assert_build_fails_and_writes_nothing(cfg)
        self.assertIn("span", msg)

    # A paper row whose paper value is a mean over seeds.

    def seed_mean_config(self, paper=None, **row):
        x = self.make_run("seed_x", noise=0.7)
        y = self.make_run("seed_y", noise=0.9)
        self.make_run("seed_z", noise=5.0)                   # a seed the paper leaves out
        mean_xy = np.mean([np.mean([np.mean(r) for r in s.values()]) for s in (x, y)])
        cfg = self.config()
        cfg["toy"]["models"].append({
            "label": "C, seed x", "run": "seed_x", "note": "seed mean",
            "paper": float(bdd.round_half_up(mean_xy, 3)) if paper is None else paper,
            "seed_mean_of": ["seed_x", "seed_y"], "seeds_not_in_paper": ["seed_z"], **row})
        return cfg

    def test_seed_mean_row_is_checked_against_the_seed_mean(self):
        self.build(self.seed_mean_config())
        man = check_dataset(self, self.out / "toy", ["a", "b"])
        self.assertEqual(man["models"], ["A", "B (our re-run)", "C, seed x"])
        self.assertIn("not in the paper's mean: 1 seed run(s)", self.printed)

    def test_seed_mean_row_with_the_wrong_paper_value_fails(self):
        self.build()
        cfg = self.seed_mean_config()
        cfg["toy"]["models"][2]["paper"] = round(cfg["toy"]["models"][2]["paper"] + 0.01, 3)
        msg = self.assert_build_fails_and_writes_nothing(cfg)
        self.assertIn("the mean of 2 seeds", msg)

    def test_seed_mean_row_needs_every_seed_score(self):
        self.build()
        cfg = self.seed_mean_config()
        path = self.root / "seed_z" / "results.json"
        res = json.loads(path.read_text())
        del res["score"]
        path.write_text(json.dumps(res))
        msg = self.assert_build_fails_and_writes_nothing(cfg)
        self.assertIn("seed_z", msg)

    # Rounding: only a row that says so may reach the paper via 4 decimals.

    def shift_stored_r(self, run, target_mean):
        """Move every stored r of `run` by the same small amount so the cohort
        mean becomes `target_mean` (well inside the stored-vs-recomputed tolerance)."""
        path = self.root / run / "results.json"
        res = json.loads(path.read_text())
        delta = target_mean - self.mean_a
        self.assertLess(abs(delta), bdd.STORED_VS_RECOMPUTED_TOL)
        for ps in res["per_subject"].values():
            ps["corr"] = [v + delta for v in ps["corr"]]
        path.write_text(json.dumps(res))

    def test_rounding_via_4_dp_needs_the_row_to_say_so(self):
        base = math.floor(self.mean_a * 1000) / 1000
        # A cohort mean of x.xxx47 rounds to x.xxx, but via x.xxx5 to x.xxx + 0.001.
        self.shift_stored_r("run_a", base + 0.00047)
        low, high = round(base, 3), round(base + 0.001, 3)
        self.build(self.config(low))
        self.assert_build_fails_and_writes_nothing(self.config(high))
        self.build(self.config(high, paper_rounding="via 4 dp"))
        msg = self.assert_build_fails_and_writes_nothing(
            self.config(low, paper_rounding="via 4 dp"))            # a stale marker
        self.assertIn("paper_rounding", msg)

    # What the builder may replace.

    def test_half_finished_swap_is_repaired(self):
        self.build()
        # Stopped after moving the old build aside, before moving the new one in.
        (self.out / "toy").rename(self.out / ".toy.old")
        (self.out / ".toy.new").mkdir()
        (self.out / ".toy.new" / "s1.json").write_text("{}")
        self.build()
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["toy"])
        check_dataset(self, self.out / "toy", ["a", "b"])
        # Stopped after moving the new build in, before deleting the old one.
        shutil.copytree(self.out / "toy", self.out / ".toy.old")
        self.build()
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["toy"])

    def test_foreign_files_are_not_replaced(self):
        self.build()
        (self.out / "toy" / "README.txt").write_text("not the builder's")
        msg = self.assert_build_fails_and_writes_nothing()
        self.assertIn("README.txt", msg)
        (self.out / "toy" / "README.txt").unlink()
        (self.out / "toy" / "sub").mkdir()
        (self.out / "toy" / "sub" / "x.json").write_text("{}")
        self.assert_build_fails_and_writes_nothing()
        self.assertNotIn("=== toy ===", self.printed, "refused only after the slow build")

    def test_json_without_a_manifest_is_not_replaced(self):
        self.out = self.tmp / "elsewhere"
        (self.out / "toy").mkdir(parents=True)
        (self.out / "toy" / "data.json").write_text("{}")
        msg = self.assert_build_fails_and_writes_nothing()
        self.assertIn("manifest.json", msg)

    def test_unwritable_destination_fails_before_the_build(self):
        self.out = self.tmp / "a_file"
        self.out.write_text("")
        with self.assertRaises(SystemExit) as cm:
            self.build()
        self.assertIn("cannot write", str(cm.exception.code))
        self.assertNotIn("=== toy ===", self.printed)
        self.assertEqual(self.out.read_text(), "")


if __name__ == "__main__":
    unittest.main()
