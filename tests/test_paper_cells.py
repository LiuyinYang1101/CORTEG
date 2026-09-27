"""The shipped paper cells reproduce the published tables.

paper_cells/ holds the per-cell result files behind the BrainTreebank tables
(3, 9, 20) and the iEEG-FM regression tables (1, 18, 19). These tests rebuild
every cell with scripts/aggregate_btb.py and scripts/aggregate_fm.py and
compare it with the paper, check the aggregation rule on synthetic runs, check
how fresh runs are compared (seed by seed, non-paper settings kept apart), and
guard the folder itself: no private paths, no arrays, no withdrawn cells,
nothing edited since release.

Everything here is CPU-only and reads ~2 MB of JSON.

Run:  python -m unittest tests.test_paper_cells -v
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import re
import shutil
import tempfile
import unittest
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CELLS = REPO / "paper_cells"
BTB = CELLS / "btb"
FM = CELLS / "fm_regression"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


agg_btb = _load("aggregate_btb")
agg_fm = _load("aggregate_fm")


def run(mod, *argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        code = mod.main(list(argv))
    return code, buf.getvalue()


def _public_btb_run(path, seed, scores, event_seed=None, merge="layerwise_gate",
                    endpoint="sentence_onset", splits=None, **settings):
    """A result file with the fields experiments/run_btb_classification.py writes.

    event_seed=None leaves it out, as the runner did before --event_seed
    existed, when --seed drew the events too."""
    args = {"merge_strategy": merge, "seed": seed, "max_per_class": 900,
            "no_pretrained": False, "endpoint": endpoint, "train_mode": "pooled",
            "per_subject_lora": True, "select_metric": "pooled",
            "model_kwargs_json": "configs/steegformer_small.json", **settings}
    out = {"cohort_mean_auroc": float(np.mean(list(scores.values()))),
           "per_subject": {s: {"folds": [v] * 4, "mean": v} for s, v in scores.items()},
           "seed": seed, "endpoint": endpoint, "merge_strategy": merge,
           "args": args}
    if event_seed is not None:
        args["event_seed"] = out["event_seed"] = event_seed
    if splits is not None:
        out["splits"] = splits
    path.write_text(json.dumps(out), encoding="utf-8")


def _shift(path, field, subject, by):
    """Add `by` to one subject's score in a result file."""
    j = json.loads(path.read_text(encoding="utf-8"))
    if field == "per_subject_auroc":
        j[field][subject] += by
    else:
        j["per_subject"][subject]["corr_mean"] += by
    path.write_text(json.dumps(j), encoding="utf-8")


# The paper's run-to-run floor: two repeats of the seed-42 Task B gate run with
# the paper code (the paper's own run is in paper_cells/btb/corteg_word_nonword).
FLOOR_REPEATS = {
    "r1": {"sub_1": 0.9251032330401427, "sub_2": 0.8129105354712708, "sub_3": 0.9145709371636647,
           "sub_4": 0.8529050686204497, "sub_5": 0.6497947073985244, "sub_6": 0.8777918536176388,
           "sub_7": 0.5567030059445917, "sub_8": 0.8109921497062138, "sub_9": 0.6013018532783856,
           "sub_10": 0.662248703298099},
    "r2": {"sub_1": 0.9331695055420207, "sub_2": 0.8100525268323728, "sub_3": 0.9050909892832059,
           "sub_4": 0.8444949294827047, "sub_5": 0.6214383935634836, "sub_6": 0.8776479549499366,
           "sub_7": 0.5675647035758686, "sub_8": 0.8088291358246006, "sub_9": 0.5880987392834052,
           "sub_10": 0.6771311154749624},
}
GATE_B_S42 = "corteg_word_nonword/results_pooled_small_s42_lora4_mni_hfa70-200_word_nonword.json"


# ─────────────────────────── BrainTreebank ──────────────────────────────────
class TestBrainTreebankCells(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cells = agg_btb.aggregate(agg_btb.load_records(str(BTB)))

    def test_every_published_cell_matches(self):
        code, out = run(agg_btb)
        self.assertEqual(code, 0, out[-3000:])
        # 256 table cells + the oracle null, Task A and B
        self.assertIn("258 cells checked against the paper: all match.", out)
        self.assertIn("Task A 0.5297, Task B 0.5284  (paper: at least 0.53", out)
        self.assertNotIn("Other cells", out)       # no paper file trips a recipe tag

    def test_the_oracle_null_is_checked(self):
        """The paper quotes the null as 'at least 0.53'; both cohort means must
        round to it, and a changed null file fails the default run."""
        with tempfile.TemporaryDirectory() as d:
            shutil.copytree(BTB, Path(d) / "btb")
            f = Path(d) / "btb" / "max_electrode_null" / "max_null_word_nonword.json"
            j = json.loads(f.read_text(encoding="utf-8"))
            for v in j["per_subject"].values():
                v["null_mean"] -= 0.02
            f.write_text(json.dumps(j), encoding="utf-8")
            saved = agg_btb.DEFAULT_CELLS
            agg_btb.DEFAULT_CELLS = str(Path(d) / "btb")
            try:
                code, out = run(agg_btb)
            finally:
                agg_btb.DEFAULT_CELLS = saved
        self.assertEqual(code, 1)
        self.assertIn("oracle null Task B: 0.51 vs paper 0.53", out)

    def test_cells_must_be_folders_and_tol_is_for_fresh_runs(self):
        for mod, good in ((agg_btb, BTB / "base_LSTM"), (agg_fm, FM / "temporal")):
            with self.subTest(script=mod.__name__):
                with self.assertRaises(SystemExit) as e:        # a typo must not pass
                    run(mod, "--cells", str(REPO / "no_such_folder"), str(good))
                self.assertNotEqual(e.exception.code, 0)
                with self.assertRaises(SystemExit) as e:        # nor a single file
                    run(mod, "--cells", str(next(good.rglob("*.json"))))
                self.assertNotEqual(e.exception.code, 0)
                with self.assertRaises(SystemExit) as e:        # paper cells: exact only
                    run(mod, "--tol", "0.001")
                self.assertNotEqual(e.exception.code, 0)

    def test_every_row_has_the_paper_seeds_and_ten_subjects(self):
        for key, _, _ in agg_btb.TABLE20:
            for task in agg_btb.TASKS:
                with self.subTest(row=key, task=task):
                    c = self.cells[(key, task)]
                    self.assertEqual(sorted(c["seeds"]),
                                     sorted(agg_btb.PAPER_SEEDS.get(key, (42,))))
                    self.assertEqual(c["subjects"], agg_btb.SUBJECTS)
                    self.assertEqual(c["warnings"], [])

    def test_three_seeds_score_one_event_set(self):
        """The paper scores one seed-42 event set under training seeds 42, 1, 2,
        so every seed of a row must carry identical test blocks."""
        recs = agg_btb.load_records(str(BTB))
        for key in agg_btb.PAPER_SEEDS:
            for task in agg_btb.TASKS:
                sigs = {r.signature for r in recs if r.row == key and r.task == task}
                self.assertEqual(len(sigs), 1, (key, task, sigs))

    def test_expected_values_round_once(self):
        """Cells just below a 3-dp boundary: the expected (printed) value rounds the
        full-precision score once; rounding its 4-dp value again would be 0.001 high."""
        _, out = run(agg_btb)
        self.assertEqual(len(agg_btb.DOUBLE_ROUNDING_TRAPS), 14)
        for (table, key, task, field), twice in agg_btb.DOUBLE_ROUNDING_TRAPS.items():
            c = self.cells[(key, task)]
            i = agg_btb.TASKS.index(task)
            if table == "T9":
                exact = c["per_subject"][field]
                want = agg_btb.EXPECTED_T9[(key, task)].split()[agg_btb.SUBJECTS.index(field)]
            else:
                exact = c["mean"] if field == "mean" else c["sd"]
                want = agg_btb.EXPECTED_T3[key][i][0 if field == "mean" else 1]
            with self.subTest(cell=(table, key, task, field)):
                self.assertEqual(want, f"{exact:.3f}")
                four = f"{exact:.4f}"
                self.assertEqual(four[-1], "5")
                self.assertEqual(str(Decimal(four).quantize(Decimal("0.001"), ROUND_HALF_UP)),
                                 twice)
                self.assertNotEqual(want, twice)
        # The published tables print the exact values; nothing is reported as a correction.
        self.assertNotIn("camera-ready", out)
        self.assertNotIn("printed", out)

    def test_aggregation_rule(self):
        """Seeds averaged within subject, then mean and ddof=1 SD across subjects;
        seed SD is the ddof=1 SD of the per-seed cohort means."""
        s42 = {"sub_1": 0.60, "sub_2": 0.70, "sub_3": 0.80}
        s1 = {"sub_1": 0.64, "sub_2": 0.66, "sub_3": 0.90}
        with tempfile.TemporaryDirectory() as d:
            _public_btb_run(Path(d) / "btb_pooled_layerwise_gate_seed42.json", 42, s42, 42)
            _public_btb_run(Path(d) / "btb_pooled_layerwise_gate_seed1.json", 1, s1, 42)
            c = agg_btb.aggregate(agg_btb.load_records(d))[("corteg_gate", "sentence_onset")]
        per = np.array([[s42[s], s1[s]] for s in sorted(s42)]).mean(axis=1)
        self.assertAlmostEqual(c["mean"], per.mean(), places=12)
        self.assertAlmostEqual(c["sd"], per.std(ddof=1), places=12)
        self.assertAlmostEqual(c["seed_sd"], np.std([np.mean(list(s42.values())),
                                                     np.mean(list(s1.values()))], ddof=1),
                               places=12)
        self.assertEqual(c["seeds"], [42, 1])
        self.assertEqual(c["warnings"], [])

    def test_a_file_without_event_seed_used_its_training_seed(self):
        """The runner version whose --seed also drew the events recorded no event
        seed; its seed-1 run was scored on event set 1, not the paper's."""
        with tempfile.TemporaryDirectory() as d:
            _public_btb_run(Path(d) / "a.json", 42, {"sub_1": 0.6, "sub_2": 0.7})
            _public_btb_run(Path(d) / "b.json", 1, {"sub_1": 0.6, "sub_2": 0.7})
            cells = agg_btb.aggregate(agg_btb.load_records(d))
        self.assertEqual(sorted(k[0] for k in cells), ["corteg_gate", "corteg_gate[es1]"])

    def test_seeds_on_different_event_sets_fail(self):
        """Seeds of one row must share their test blocks; a fresh run where they
        do not fails, even when each seed matches its reference seed."""
        ref = agg_btb.aggregate(agg_btb.load_records(str(BTB / "corteg_sentence_onset")))
        c = ref[("corteg_gate", "sentence_onset")]
        with tempfile.TemporaryDirectory() as d:
            for sd, blocks in ((42, [{"subj": "sub_1", "fold": 0, "n_test": 100}]),
                               (1, [{"subj": "sub_1", "fold": 0, "n_test": 120}])):
                _public_btb_run(Path(d) / f"s{sd}.json", sd, c["per_seed_subject"][sd], 42,
                                splits=blocks)
            cells = agg_btb.aggregate(agg_btb.load_records(d))
            self.assertTrue(any("different event sets" in w
                                for w in cells[("corteg_gate", "sentence_onset")]["warnings"]))
            code, out = run(agg_btb, "--cells", d)
        self.assertEqual(code, 1, out[-2000:])
        self.assertRegex(out, r"FAILED:\n(.*\n)*.*were scored on different event sets")

    def test_non_paper_settings_never_enter_a_published_row(self):
        with tempfile.TemporaryDirectory() as d:
            _public_btb_run(Path(d) / "a.json", 42, {"sub_1": 0.6, "sub_2": 0.7}, 42)
            _public_btb_run(Path(d) / "b.json", 42, {"sub_1": 0.9, "sub_2": 0.9}, 42,
                            per_subject_lora=False)
            _public_btb_run(Path(d) / "c.json", 42, {"sub_1": 0.9, "sub_2": 0.9}, 7)
            cells = agg_btb.aggregate(agg_btb.load_records(d))
        self.assertEqual(sorted(k[0] for k in cells),
                         ["corteg_gate", "corteg_gate[es7]", "corteg_gate[sharedlora]"])
        self.assertAlmostEqual(cells[("corteg_gate", "sentence_onset")]["mean"], 0.65)

    def test_recipe_changes_and_truncated_runs_get_their_own_row(self):
        scores = {"sub_1": 0.6, "sub_2": 0.7}
        cases = {
            "n_folds": ({"n_folds": 1}, "corteg_gate[n_folds=1]"),
            "epochs": ({"epochs": 1}, "corteg_gate[epochs=1]"),
            "lr": ({"lr": 1e-2}, "corteg_gate[lr=0.01]"),
            "variant": ({"steegformer_variant": "base",
                         "model_kwargs_json": "configs/steegformer_base.json"},
                        "corteg_gate[steegformer_variant=base,model_kwargs_json="
                        "steegformer_base.json]"),
            "window": ({"win_sec": 0.5, "pre_sec": -0.5},
                       "corteg_gate[win_sec=0.5,pre_sec=-0.5]"),
            "band": ({"hga_low": 60.0}, "corteg_gate[hga_low=60]"),
            "gate": ({"layerwise_gate_act": "sigmoid"}, "corteg_gate[layerwise_gate_act=sigmoid]"),
            "lora": ({"lora_r": 8}, "corteg_gate[lora_r=8]"),
            "smoke": ({"max_per_class": 50, "epochs": 1}, "corteg_gate[n50,epochs=1]"),
            "randinit_merge": ({"no_pretrained": True, "max_per_class": 50},
                               "corteg_randinit[average,n50]"),
            "clip": ({"max_norm": 0.1}, "corteg_gate[max_norm=0.1]"),
            "min_lr": ({"min_lr": 1e-3}, "corteg_gate[min_lr=0.001]"),
            "warmup": ({"warmup_epochs": 0}, "corteg_gate[warmup_epochs=0]"),
            "warmup_default": ({"warmup_epochs": 6, "min_lr": 1e-5, "max_norm": 1.0},
                               "corteg_gate"),
            "warmup_follows_epochs": ({"epochs": 1, "warmup_epochs": 1},
                                      "corteg_gate[epochs=1]"),
            "subject_order": ({"subjects": [f"sub_{i}" for i in range(1, 11)]},
                              "corteg_gate[subj1-2-3-4-5-6-7-8-9-10]"),
            "paper_order": ({"subjects": list(agg_btb.PAPER_SUBJECT_ORDER)}, "corteg_gate"),
            "subset": ({"subjects": ["sub_9", "sub_2"]}, "corteg_gate[subj9-2]"),
            "per_subject_set": ({"train_mode": "per_subject",
                                 "subjects": [f"sub_{i}" for i in range(10, 0, -1)]},
                                "corteg_gate[per_subject]"),
            "trial": ({"trial": "trial002"}, "corteg_gate[trial=trial002]"),
        }
        for name, (settings, row) in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as d:
                merge = "average" if name == "randinit_merge" else "layerwise_gate"
                _public_btb_run(Path(d) / "a.json", 42, scores, 42, merge=merge, **settings)
                self.assertEqual([k[0] for k in agg_btb.aggregate(agg_btb.load_records(d))],
                                 [row])

    def test_other_runners_recipe_changes_are_tagged(self):
        base = json.loads((BTB / "base_HiLoFuseNet" / "results_pooled_sentence_onset_s42.json")
                          .read_text(encoding="utf-8"))
        base["hp"]["epochs"] = 2
        base["folds"] = 2
        popt = json.loads((BTB / "popt_finetune_sentence_onset" /
                           "results_popt_lora_sentence_onset_seed42.json").read_text(encoding="utf-8"))
        popt["config"].update(folds=1, epochs=1, max_per_class=50)
        fm = {"fm": "brainbert", "arm": "single_elec_max", "endpoint": "sentence_onset",
              "per_subject": {"sub_1": {"auroc": 0.6}},
              "args": {"seed": 42, "n_folds": 1, "max_per_class": 900, "event_seed": 42}}
        got = {r.row for name, d in (("b", base), ("p", popt), ("f", fm))
               for r in agg_btb.classify(name, d)}
        self.assertEqual(got, {"hilofusenet[n_folds=2,epochs=2]",
                               "popt_lora[n50,folds=1,epochs=1]",
                               "bb_single_max[n_folds=1]"})

    def test_popt_and_frozen_settings_the_paper_runs_record(self):
        """PopT: validation interval and gradient clipping (the paper config records
        eval_every 2, max_norm 1.0) and a forced --trial; the frozen runner's
        --trial. A subject subset is the same cell for these per-subject fits."""
        src = json.loads((BTB / "popt_finetune_word_nonword" /
                          "results_popt_lora_word_nonword_seed42.json").read_text(encoding="utf-8"))
        cases = {
            "eval_every": ({"eval_every": 1}, None, "popt_lora[eval_every=1]"),
            "clip": ({"max_norm": 0.1}, None, "popt_lora[max_norm=0.1]"),
            "trial": ({"trial": "trial000"}, None, "popt_lora[trial=trial000]"),
            "cpu": ({"device": "cpu"}, [], "popt_lora"),
            "subset": ({"subjects": ["sub_1", "sub_2"]}, ["sub_1+sub_2"], "popt_lora"),
            # a setting the aggregator does not know, flagged by the runner itself
            "runner_flag": ({}, ["new_flag3"], "popt_lora[new_flag3]"),
        }
        for name, (cfg, name_tags, row) in cases.items():
            with self.subTest(case=name):
                d = json.loads(json.dumps(src))
                d["config"].update(cfg)
                if name_tags is not None:
                    d["name_tags"] = name_tags
                self.assertEqual([r.row for r in agg_btb.classify("p", d)], [row])
        fm = {"fm": "brainbert", "arm": "single_elec_max", "endpoint": "sentence_onset",
              "per_subject": {"sub_1": {"auroc": 0.6}}, "name_tags": ["trial000"],
              "args": {"seed": 42, "n_folds": 4, "val_frac": 0.0, "max_per_class": 900,
                       "event_seed": 42, "trial": "trial000"}}
        self.assertEqual([r.row for r in agg_btb.classify("f", fm)],
                         ["bb_single_max[trial=trial000]"])

    def test_decoder_training_mode_and_subjects_get_their_own_row(self):
        """experiments/run_btb_baselines.py records train_mode and subjects; the paper
        decoder files record neither (pooled, paper order). A per-subject run or a
        subject subset is another experiment: it must not be scored as the pooled
        row, nor collide with it when written into the same folder."""
        def public(seed, mode="pooled", subjects=None, shift=0.0):
            name = f"results_pooled_sentence_onset_s{seed}.json"
            src = json.loads((BTB / "base_HiLoFuseNet" / name).read_text(encoding="utf-8"))
            subjects = list(agg_btb.PAPER_SUBJECT_ORDER) if subjects is None else subjects
            return {"decoder": "HiLoFuseNet", "endpoint": "sentence_onset", "seed": seed,
                    "event_seed": 42, "train_mode": mode, "neg_mode": None,
                    "nonpaper_settings": [], "subjects": subjects,
                    "per_subject_auroc": {s: src["per_subject_auroc"][s] + shift
                                          for s in subjects},
                    "folds": 4, "val_frac": 0.15, "splits": src["splits"], "loss": "bce",
                    "hp": dict(src["hp"], patience=10),
                    "features": {"win_sec": 1.5, "pre_sec": 0.0, "hga_band": [70.0, 200.0],
                                 "max_per_class": 900, "neg_mode": None}}
        rows = {name: [r.row for r in agg_btb.classify("x", d)] for name, d in {
            "pooled": public(42),
            "per_subject": public(42, "per_subject", shift=0.02),
            "subset": public(42, subjects=["sub_1", "sub_2"]),
            "order": public(42, subjects=[f"sub_{i}" for i in range(1, 11)]),
        }.items()}
        self.assertEqual(rows, {"pooled": ["hilofusenet"],
                                "per_subject": ["hilofusenet[per_subject]"],
                                "subset": ["hilofusenet[subj1-2]"],
                                "order": ["hilofusenet[subj1-2-3-4-5-6-7-8-9-10]"]})
        with tempfile.TemporaryDirectory() as d:
            # one folder, as the runner writes it: the pooled seeds, the per-subject
            # seeds (scores moved by +0.02) and a two-subject smoke run
            for seed in (42, 1, 2):
                for mode, shift in (("pooled", 0.0), ("per_subject", 0.02)):
                    (Path(d) / f"results_{mode}_sentence_onset_s{seed}.json").write_text(
                        json.dumps(public(seed, mode, shift=shift)), encoding="utf-8")
            (Path(d) / "results_pooled_sentence_onset_s42_subj1-2.json").write_text(
                json.dumps(public(42, subjects=["sub_1", "sub_2"])), encoding="utf-8")
            code, out = run(agg_btb, "--cells", d)
        self.assertEqual(code, 0, out[-3000:])
        self.assertRegex(out, r"\nHiLoFuseNet +0\.533\+-0\.022 ")
        self.assertRegex(out, r"  hilofusenet\[per_subject\] +Task A  0\.5530\+-0\.0220")
        self.assertIn("  hilofusenet[subj1-2] ", out)

    def test_the_same_cell_in_two_files_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            _public_btb_run(Path(d) / "a.json", 42, {"sub_1": 0.6, "sub_2": 0.7}, 42)
            _public_btb_run(Path(d) / "b.json", 42, {"sub_1": 0.6, "sub_2": 0.7}, 42)
            with self.assertRaises(SystemExit):
                agg_btb.aggregate(agg_btb.load_records(d))

    def test_fresh_runs_are_compared_seed_by_seed(self):
        with tempfile.TemporaryDirectory() as d:
            shutil.copytree(BTB / "corteg_sentence_onset", Path(d) / "same")
            code, out = run(agg_btb, "--cells", str(Path(d) / "same"))
            self.assertEqual(code, 0, out[-2000:])
            self.assertRegex(out, r"corteg_gate +Task A seed 42 +n=10 +cohort d=\+0\.0000")
            self.assertIn("all compared cells and seeds are within tolerance", out)

            # Move one subject of one seed by 0.05: beyond the per-subject tolerance
            # (1.5 x the largest per-subject difference between repeated runs,
            # 0.0284), though the cohort mean moves only 0.005, inside its own.
            _shift(Path(d) / "same" / "results_pooled_small_s1_lora4_mni_hfa70-200.json",
                   "per_subject_auroc", "sub_5", 0.05)
            code, out = run(agg_btb, "--cells", str(Path(d) / "same"))
            self.assertEqual(code, 1)
            self.assertIn("corteg_gate Task A seed 1: seed-matched", out)

    def test_a_single_seed_rerun_is_checked_seed_by_seed(self):
        """A seed-42-only rerun of a three-seed row is not scored against the
        three-seed mean (seed 42 alone gives 0.6499 for CORTEG Task A, the row
        0.6376); it passes or fails on the seed-matched comparison alone."""
        with tempfile.TemporaryDirectory() as d:
            for f in (sorted((BTB / "corteg_sentence_onset").glob("*_s42_*.json"))
                      + sorted((BTB / "corteg_word_nonword").glob("*_s42_*.json"))
                      + sorted((BTB / "base_HiLoFuseNet").glob("*_s42.json"))):
                shutil.copy(f, d)
            code, out = run(agg_btb, "--cells", d)
            self.assertEqual(code, 0, out[-3000:])
            self.assertIn("0.6499+-0.0752", out)
            self.assertIn("[not scored: seeds 42, paper 42,1,2]", out)
            self.assertIn("6 seeds compared with the same seed", out)

            gate_a = Path(d) / "results_pooled_small_s42_lora4_mni_hfa70-200.json"
            _shift(gate_a, "per_subject_auroc", "sub_5", 0.05)
            code, out = run(agg_btb, "--cells", d)
            self.assertEqual(code, 1)
            self.assertIn("corteg_gate Task A seed 42: seed-matched", out)

            _shift(gate_a, "per_subject_auroc", "sub_5", -0.05)          # undo
            for s in agg_btb.SUBJECTS:                                   # cohort-wide shift
                _shift(gate_a, "per_subject_auroc", s, 0.015)
            code, out = run(agg_btb, "--cells", d)
            self.assertEqual(code, 1)
            self.assertIn("corteg_gate Task A seed 42: seed-matched cohort d=+0.0150", out)

    def test_a_failure_that_looks_like_another_seed_is_marked(self):
        """The seed-matched check presumes the paper's random stream. A run whose
        seed 1 is really another seed (here: the paper's seed 2) fails, and the
        output says it lies within the paper's own seed-to-seed spread; where that
        spread is inside the tolerances, the output says the check cannot tell."""
        with tempfile.TemporaryDirectory() as d:
            src = BTB / "corteg_sentence_onset"
            for f in src.glob("*_s42_lora4_mni_hfa70-200.json"):
                shutil.copy(f, d)
            s2 = json.loads((src / "results_pooled_small_s2_lora4_mni_hfa70-200.json")
                            .read_text(encoding="utf-8"))
            s2["seed"] = 1
            (Path(d) / "s1.json").write_text(json.dumps(s2), encoding="utf-8")
            for f in (BTB / "base_LSTM").glob("*word_nonword_s42.json"):
                shutil.copy(f, d)
            code, out = run(agg_btb, "--cells", d)
        self.assertEqual(code, 1)
        self.assertRegex(out, r"corteg_gate +Task A: the reference seeds 42/1/2 differ from "
                              r"each other by up to cohort 0\.0268")
        self.assertRegex(out, r"corteg_gate +Task A seed 1 .*<-- outside tolerance; within "
                              r"the reference's seed-to-seed spread")
        self.assertIn("1 of them is no further from their reference seed", out)
        self.assertRegex(out, r"lstm +Task B: .*\(inside the tolerances: this check cannot "
                              r"tell another seed from the same one\)")

    def test_the_papers_repeat_runs_pass_against_each_other(self):
        """The defaults must accept the three identical-config runs they were
        measured from, in every pairing."""
        with tempfile.TemporaryDirectory() as d:
            for name, scores in FLOOR_REPEATS.items():
                (Path(d) / name).mkdir()
                _public_btb_run(Path(d) / name / "run.json", 42, scores, 42,
                                endpoint="word_nonword")
            (Path(d) / "paper").mkdir()
            shutil.copy(BTB / GATE_B_S42, Path(d) / "paper")
            for fresh, ref in (("r1", "paper"), ("r2", "paper"), ("r1", "r2"), ("r2", "r1"),
                               ("paper", "r1"), ("paper", "r2")):
                with self.subTest(fresh=fresh, reference=ref):
                    code, out = run(agg_btb, "--cells", str(Path(d) / fresh),
                                    "--reference", str(Path(d) / ref))
                    self.assertEqual(code, 0, out[-1500:])
                    self.assertIn("1 seeds compared with the same seed", out)

    def test_tolerances_derive_from_the_floor(self):
        self.assertEqual((agg_btb.TOL, agg_btb.TOL_SUBJECT, agg_btb.TOL_SUBJECT_MAX),
                         (0.0111, 0.0148, 0.0426))
        for name, v in agg_btb.FLOOR.items():
            self.assertLess(v, agg_btb._floor_tol(name))

    def test_a_missing_reference_is_an_error(self):
        with self.assertRaises(SystemExit) as e:
            run(agg_btb, "--cells", str(BTB / "corteg_sentence_onset"),
                "--reference", str(REPO / "no_such_folder"))
        self.assertNotEqual(e.exception.code, 0)

    def test_nothing_compared_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as d:
            _public_btb_run(Path(d) / "smoke.json", 42, {"sub_1": 0.6, "sub_2": 0.7}, 42,
                            max_per_class=50, epochs=1)
            (Path(d) / "summary.json").write_text('{"mean": 0.6}', encoding="utf-8")
            code, out = run(agg_btb, "--cells", d)
        self.assertEqual(code, 1)
        self.assertIn("NOTHING VERIFIED", out)
        self.assertIn("corteg_gate[n50,epochs=1]", out)

    def test_rows_without_a_released_runner_are_not_missing(self):
        with tempfile.TemporaryDirectory() as d:
            shutil.copy(BTB / GATE_B_S42, d)
            _, out = run(agg_btb, "--cells", d)
        self.assertIn("18 from runners not in this release", out)
        self.assertEqual(out.count("(runner not released)"), len(agg_btb.NOT_RELEASED))


# ─────────────────────────── iEEG-FM regression ─────────────────────────────
def _copy_seed42_stanford(dst):
    for src in sorted(FM.glob("*/Stanford/*/*/seed42")):
        shutil.copytree(src, Path(dst) / src.relative_to(FM))


class TestFmRegressionCells(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        recs, cls.incomplete = agg_fm.load_records(str(FM))
        cls.cells = agg_fm.aggregate(recs)

    def test_every_published_cell_matches(self):
        code, out = run(agg_fm)
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("78 cells checked against the paper: all match.", out)
        self.assertIn("SD > mean in 11/14 cells", out)
        self.assertNotIn("Other cells", out)
        self.assertNotIn("camera-ready", out)

    def test_table18_picks_the_paper_regimes(self):
        for (fm, ad), per_ds in agg_fm.EXPECTED_T18.items():
            for ds, (_, _, three, regime) in per_ds.items():
                with self.subTest(cell=(fm, ad, ds)):
                    c = agg_fm.best_regime(self.cells, fm, ad, ds)
                    self.assertEqual(c["key"][3], regime)
                    self.assertEqual(c["n_seeds"] > 1, three)

    def test_subject_counts_and_seeds(self):
        for key, c in self.cells.items():
            with self.subTest(cell=key):
                self.assertEqual(len(c["subjects"]), agg_fm.N_SUBJECTS[key[1]])
                self.assertFalse(c["ragged"])
                self.assertEqual(c["seeds"], list(agg_fm.paper_seeds(key)))

    def test_withdrawn_cells_are_not_shipped(self):
        self.assertFalse((FM / "ft" / "Stanford" / "brainbert" / "pooled").exists())
        self.assertFalse((FM / "ft" / "Ghent" / "brainbert" / "loo").exists())
        for ds, rg, fm, ad in agg_fm.WITHDRAWN:
            self.assertNotIn((ad, ds, fm, rg), self.cells)
            self.assertIsNone(agg_fm.EXPECTED_T19[(ds, rg)][agg_fm.T19_COLS.index((fm, ad))])
        self.assertEqual(self.incomplete, [])

    def test_ghent_loo_popt_ft_is_rounded_once(self):
        c = self.cells[("ft", "Ghent", "popt", "loo")]
        self.assertEqual(f"{c['mean']:.4f}", "0.0125")      # a 4-dp detour would give 0.013
        self.assertEqual(f"{c['mean']:.3f}", "0.012")       # ... the exact value is 0.012
        self.assertEqual(agg_fm.EXPECTED_T19[("Ghent", "loo")][3], "0.012")

    def test_loo_summaries_agree_with_their_folds(self):
        """_loo_done.json is what counts; its per-subject r must be the r its
        ft_<subject> run recorded, and every held-out subject must be there."""
        for done in sorted(FM.glob("*/*/*/loo/seed42/_loo_done.json")):
            with self.subTest(cell=done.parent.relative_to(FM).as_posix()):
                summary = json.loads(done.read_text(encoding="utf-8"))["per_subject"]
                folds = sorted(done.parent.glob("ft_*/results_persub.json"))
                self.assertEqual(len(folds), len(summary))
                for f in folds:
                    subj = f.parent.name[len("ft_"):]
                    got = json.loads(f.read_text(encoding="utf-8"))["per_subject"][subj]
                    self.assertEqual(got["corr_mean"], summary[subj]["corr_mean"])

    def test_loo_folds_without_the_done_marker_are_not_counted(self):
        with tempfile.TemporaryDirectory() as d:
            dst = Path(d) / "probe" / "Stanford" / "popt" / "loo" / "seed42"
            shutil.copytree(FM / "probe" / "Stanford" / "popt" / "loo" / "seed42", dst)
            (dst / "_loo_done.json").unlink()
            recs, incomplete = agg_fm.load_records(d)
        self.assertEqual(recs, [])
        self.assertEqual(len(incomplete), 1)
        self.assertIn("9 LOO folds", incomplete[0])

    def test_fresh_runs_are_compared_seed_by_seed(self):
        with tempfile.TemporaryDirectory() as d:
            shutil.copytree(FM / "temporal" / "Stanford", Path(d) / "temporal" / "Stanford")
            code, out = run(agg_fm, "--cells", d)
            self.assertEqual(code, 0, out[-2000:])
            self.assertRegex(out, r"temporal/Stanford/popt/per_subject +seed 42 +n=9 +"
                                  r"cohort d=\+0\.0000")

    def test_a_single_seed_rerun_is_checked_seed_by_seed(self):
        """Seed 42 of the three-seed cells is not scored against the three-seed
        means (e.g. Stanford pooled PopT probe: 0.047 at seed 42, 0.040 over three),
        nor Table 18 against a paper cell chosen from other regimes."""
        with tempfile.TemporaryDirectory() as d:
            _copy_seed42_stanford(d)
            code, out = run(agg_fm, "--cells", d)
            self.assertEqual(code, 0, out[-3000:])
            self.assertIn("T19 Stanford pooled PopT probe  [not scored: pooled seeds 42, "
                          "paper 42,0,1]", out)
            self.assertIn("NOTE T18 PopT probe finger: best regime here per-subject, paper LOO",
                          out)
            self.assertIn("17 seeds compared with the same seed", out)

            f = Path(d) / "probe" / "Stanford" / "popt" / "per_subject" / "seed42" / \
                "results_persub.json"
            _shift(f, "corr_mean", "cc", 0.05)
            code, out = run(agg_fm, "--cells", d)
            self.assertEqual(code, 1)
            self.assertIn("probe/Stanford/popt/per_subject seed 42: seed-matched", out)

    def test_table18_needs_every_paper_regime(self):
        with tempfile.TemporaryDirectory() as d:
            src = FM / "probe" / "Stanford" / "popt" / "per_subject"
            shutil.copytree(src, Path(d) / src.relative_to(FM))
            code, out = run(agg_fm, "--cells", d)
        self.assertEqual(code, 0, out[-2000:])
        self.assertIn("T18 PopT probe finger  [not scored: no LOO run; no pooled run]", out)
        self.assertIn("1 of ", out)                  # only the Table 19 cell is scored

    def test_the_release_grid_scores_every_stanford_cell(self):
        """scripts/table1_ieeg_fm_stanford.sh runs every Stanford cell here except
        Brant's unreported pooled head. That grid must make every Stanford Table 18
        and Table 1 cell scorable: Brant's Table 18 cell is its per-subject run."""
        with tempfile.TemporaryDirectory() as d:
            for src in sorted(FM.glob("*/Stanford/*/*/seed*")):
                rel = src.relative_to(FM)
                if rel.parts[0] == "brant" and rel.parts[3] == "pooled":
                    continue
                shutil.copytree(src, Path(d) / rel)
            code, out = run(agg_fm, "--cells", d)
        self.assertEqual(code, 0, out[-3000:])
        t18_t1 = out[out.index("Table 18"):out.index("Seed-matched")]
        self.assertNotIn("finger  [not scored", t18_t1)
        self.assertIn("31 of 52 published cells compared with the paper", out)
        self.assertIn("30 seeds compared with the same seed", out)

    def test_brants_pooled_head_is_shown_but_never_chosen(self):
        _, out = run(agg_fm)
        self.assertIn("Brant pooled head (not reported in the paper; not a Table 18 "
                      "candidate): finger 0.001+-0.018, audio 0.014+-0.035", out)
        with tempfile.TemporaryDirectory() as d:
            src = FM / "brant" / "Stanford"
            shutil.copytree(src, Path(d) / "brant" / "Stanford")
            f = Path(d) / "brant" / "Stanford" / "brant" / "pooled" / "seed42" / \
                "results_pooled.json"
            for s in json.loads(f.read_text(encoding="utf-8"))["per_subject"]:
                _shift(f, "corr_mean", s, 0.2)
            code, out = run(agg_fm, "--cells", d)
        self.assertEqual(code, 1, out[-2000:])     # the moved pooled seed fails seed-matching
        self.assertRegex(out, r"Brant +frozen probe \(6 s context\) +"
                              r"0\.028\+-0\.031 \[per-subject\]")
        self.assertIn("NOTE T18 Brant brant finger: the pooled head (0.201) beats the "
                      "per-subject cell here", out)

    def test_seed_count_is_printed_as_found(self):
        subs = ["bp", "cc", "ht", "jc", "jp", "mv", "wc", "wm", "zt"]
        recs = [agg_fm.Record("probe", "Stanford", "popt", "pooled", sd,
                              {s: 0.1 + 0.01 * i for i, s in enumerate(subs)}, "x")
                for sd in (42, 0)]
        lines = []
        agg_fm.print_table18(agg_fm.aggregate(recs), agg_fm.Checker(0.005), lines.append)
        self.assertIn("(2 seeds) [pooled]", "\n".join(lines))

    def test_non_paper_settings_are_kept_apart(self):
        """Runner folder tags and recipe changes never land in a paper cell."""
        cases = {
            "bb_pool": ({"bb_pool": "last10"}, "probe_last10"),
            "smoke": ({"max_windows": 300, "epochs": 1}, "probe[epochs=1,smoke]"),
            "reref": ({"reref": "car"}, "probe[reref=car]"),
            "batch": ({"batch_size": 16}, "probe[batch_size=16]"),
        }
        for name, (settings, adaptation) in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as d:
                dst = Path(d) / "x"
                shutil.copytree(FM / "probe" / "Stanford" / "brainbert" / "pooled", dst)
                for f in dst.rglob("results_*.json"):
                    j = json.loads(f.read_text(encoding="utf-8"))
                    j["args"].update(settings)
                    f.write_text(json.dumps(j), encoding="utf-8")
                recs, _ = agg_fm.load_records(d)
                self.assertEqual({r.key for r in recs},
                                 {(adaptation, "Stanford", "brainbert", "pooled")})
        ft = json.loads((FM / "ft" / "Stanford" / "popt" / "pooled" / "seed42" /
                         "results_pooled.json").read_text(encoding="utf-8"))
        ft["args"]["unfreeze_last_n"] = 4
        self.assertEqual(agg_fm.classify("results_pooled.json", ft).key[0],
                         "ft[unfreeze_last_n=4]")
        tmp = json.loads((FM / "temporal" / "Stanford" / "popt" / "pooled" / "seed42" /
                          "results_pooled.json").read_text(encoding="utf-8"))
        tmp["args"]["seq_len"] = 16
        self.assertEqual(agg_fm.classify("results_pooled.json", tmp).key[0],
                         "temporal[seq_len=16]")
        brant = json.loads((FM / "brant" / "Stanford" / "brant" / "pooled" / "seed42" /
                            "results_pooled.json").read_text(encoding="utf-8"))
        brant["args"]["max_anchors"] = 3
        self.assertEqual(agg_fm.classify("results_pooled.json", brant).key[0], "brant[smoke]")

    def test_temporal_loo_and_partial_cohorts_stay_out_of_table18(self):
        """The temporal head ran pooled and per-subject only; a LOO run of it is an
        other cell. A cell without the whole cohort never wins a Table 18 regime."""
        with tempfile.TemporaryDirectory() as d:
            src = FM / "temporal" / "Stanford" / "popt"
            shutil.copytree(src, Path(d) / src.relative_to(FM))
            loo = Path(d) / "temporal" / "Stanford" / "popt" / "loo" / "seed42"
            loo.mkdir(parents=True)
            j = json.loads((src / "per_subject" / "seed42" / "results_persub.json")
                           .read_text(encoding="utf-8"))
            (loo / "ft_bp").mkdir()
            (loo / "ft_bp" / "results_persub.json").write_text(
                json.dumps({**j, "train_mode": "loo"}), encoding="utf-8")
            for v in j["per_subject"].values():
                v["corr_mean"] = 0.5
            (loo / "_loo_done.json").write_text(
                json.dumps({"fm": "popt", "dataset": "Stanford", "train_mode": "loo",
                            "mode": "probe", "head": "temporal",
                            "per_subject": j["per_subject"]}), encoding="utf-8")
            # and a three-subject per-subject probe run that would otherwise win
            part = Path(d) / "probe" / "Stanford" / "popt" / "loo" / "seed42"
            part.mkdir(parents=True)
            (part / "_loo_done.json").write_text(json.dumps(
                {"fm": "popt", "dataset": "Stanford", "train_mode": "loo", "mode": "probe",
                 "head": "linear", "per_subject": {s: {"corr_mean": 0.5}
                                                   for s in ("bp", "cc", "ht")}}),
                encoding="utf-8")
            recs, _ = agg_fm.load_records(d)
            cells = agg_fm.aggregate(recs)
            self.assertEqual(agg_fm.best_regime(cells, "popt", "temporal", "Stanford")["key"][3],
                             "per_subject")
            self.assertIsNone(agg_fm.best_regime(cells, "popt", "probe", "Stanford"))
            _, out = run(agg_fm, "--cells", d)
        self.assertRegex(out, r"Other cells.*\n(.*\n)*  temporal/Stanford/popt/loo ")
        self.assertIn("WARNING probe/Stanford/popt/loo: 3 subjects", out)

    def test_a_missing_reference_is_an_error(self):
        with self.assertRaises(SystemExit) as e:
            run(agg_fm, "--cells", str(FM / "temporal"), "--reference", str(REPO / "nope"))
        self.assertNotEqual(e.exception.code, 0)

    def test_nothing_compared_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as d:
            dst = Path(d) / "x"
            shutil.copytree(FM / "probe" / "Stanford" / "brainbert" / "pooled" / "seed42", dst)
            f = dst / "results_pooled.json"
            j = json.loads(f.read_text(encoding="utf-8"))
            j["args"]["max_windows"] = 300
            f.write_text(json.dumps(j), encoding="utf-8")
            code, out = run(agg_fm, "--cells", d)
        self.assertEqual(code, 1)
        self.assertIn("NOTHING VERIFIED", out)

    def test_seed_average_is_within_subject(self):
        c = self.cells[("probe", "Ghent", "brainbert", "per_subject")]
        M = np.array([[c["per_seed_subject"][sd][s] for s in c["subjects"]]
                      for sd in c["seeds"]])
        self.assertAlmostEqual(c["sd"], M.mean(axis=0).std(ddof=1), places=12)
        self.assertEqual(f"{c['mean']:.3f}+-{c['sd']:.3f}", "0.052+-0.067")
        # ddof matters at 3 dp: the population SD would print 0.065.
        self.assertEqual(f"{M.mean(axis=0).std(ddof=0):.3f}", "0.065")


# ─────────────────────────── the folder itself ──────────────────────────────
class TestPaperCellsFolder(unittest.TestCase):
    FILES = sorted(p for p in CELLS.rglob("*") if p.is_file())

    def test_only_json_plus_manifest_and_checksums(self):
        extra = [p.relative_to(CELLS).as_posix() for p in self.FILES
                 if p.suffix != ".json" and p.name not in ("MANIFEST.md", "SHA256SUMS")]
        self.assertEqual(extra, [])

    def test_no_private_paths(self):
        bad = re.compile(r"/home/|/lustre|/scratch/|/media/|/tmp/|vsc3\d|liuyin")
        for p in self.FILES:
            with self.subTest(file=p.relative_to(CELLS).as_posix()):
                self.assertIsNone(bad.search(p.read_text(encoding="utf-8")))

    def test_no_superseded_or_control_runs(self):
        bad = re.compile(r"superseded|contaminated|mse_|lr1e-3|_ctrl|shortsilence|bce_tune|"
                         r"preonset|mniset|matched|persubj|sweep|noise_floor")
        for p in self.FILES:
            self.assertIsNone(bad.search(p.relative_to(CELLS).as_posix()), p)

    def test_every_file_parses(self):
        for p in self.FILES:
            if p.suffix == ".json":
                with self.subTest(file=p.name):
                    self.assertIsInstance(json.loads(p.read_text(encoding="utf-8")), dict)

    def test_small(self):
        self.assertLess(sum(p.stat().st_size for p in self.FILES), 5_000_000)

    def test_files_match_their_checksums(self):
        """The cells are a release artifact: any edit, even to a config field,
        must be deliberate and come with a new SHA256SUMS."""
        listed = {}
        for line in (CELLS / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            digest, name = line.split(maxsplit=1)
            listed[name] = digest
        present = {p.relative_to(CELLS).as_posix() for p in self.FILES if p.suffix == ".json"}
        self.assertEqual(set(listed), present)
        for name, digest in listed.items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((CELLS / name).read_bytes()).hexdigest(), digest)

    def test_manifest_names_every_cell_directory(self):
        text = (CELLS / "MANIFEST.md").read_text(encoding="utf-8")
        for p in sorted(BTB.iterdir()):
            if p.is_dir():   # the manifest writes the two tasks as <task>
                name = re.sub(r"sentence_onset|word_nonword", "<task>", p.name)
                self.assertTrue(name in text, f"MANIFEST.md does not describe btb/{p.name}")
        for p in sorted(FM.iterdir()):
            if p.is_dir():
                self.assertTrue(f"`{p.name}/`" in text,
                                f"MANIFEST.md does not describe fm_regression/{p.name}")

    def test_manifest_marks_rows_without_a_released_runner(self):
        text = (CELLS / "MANIFEST.md").read_text(encoding="utf-8")
        labels = {k: lab for k, lab, _ in agg_btb.TABLE20}
        rows = [line for line in text.splitlines() if line.startswith("| ")]
        for key in agg_btb.NOT_RELEASED:
            parts = [p.strip() for p in labels[key].split(",")]
            with self.subTest(row=key):
                self.assertTrue(any(all(p in r for p in parts) and "not released" in r
                                    for r in rows), key)
        self.assertNotIn("3fd5551", text)                 # a private commit hash


if __name__ == "__main__":
    unittest.main()
