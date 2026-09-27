"""Tests for the from-scratch BrainTreebank baselines.

Covers the model classes (``models/baselines.py``), the runner
(``experiments/run_btb_baselines.py``) and its script. Everything runs on CPU
on synthetic data, or on the paper-run files in ``paper_cells/btb``, in well
under a minute; no BrainTreebank download is needed.

Run:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "table3_btb_baselines.sh"
CELLS = REPO / "paper_cells" / "btb"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_release_integrity import script_flags  # noqa: E402

import torch  # noqa: E402

from experiments import run_btb_baselines as rb  # noqa: E402
from models.baselines import CNN_LSTM, LSTM, HiLoFuseNet  # noqa: E402


def _n_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def _args(*argv):
    return rb.build_parser().parse_args(list(argv))


class TestBaselineModels(unittest.TestCase):
    """The classes must stay the ones that produced the published numbers."""

    def test_parameter_counts_are_unchanged(self):
        """Counts at the paper's settings, recorded from the original file.

        With one output (the BrainTreebank logit): C=64 gives 333,825 / 556,961
        and C=191 is sub_7, the widest BrainTreebank subject and so the pooled
        model's width. With Table 1's five finger outputs, C=64 gives 334,341 /
        557,477, its 334 K / 557 K. An edit to any layer changes a count.
        """
        want = {(64, 1): (333825, 556961, 428289), (191, 1): (337889, 593537, 688385),
                (64, 5): (334341, 557477, 428805)}
        for (C, out), (hilo, cnn, lstm) in want.items():
            self.assertEqual(_n_params(HiLoFuseNet(C=C, F=2, lstm_hidden=256, D=16,
                                                   output_size=out, dropout_prob=0.5)), hilo)
            self.assertEqual(_n_params(CNN_LSTM(input_size=C, output_size=out,
                                                dropout_prob=0.5)), cnn)
            self.assertEqual(_n_params(LSTM(input_size=C * 2, hidden_size=256,
                                            output_size=out, dropout_prob=0.5)), lstm)

    def test_runner_builds_the_paper_configuration(self):
        """build() must pass hidden/D/dropout through, with a one-logit head."""
        dev = torch.device("cpu")
        for dec, cls in (("HiLoFuseNet", HiLoFuseNet), ("CNN_LSTM", CNN_LSTM),
                         ("LSTM", LSTM)):
            m = rb.build(dec, 64, 2, dev)
            self.assertIsInstance(m, cls)
        self.assertEqual(_n_params(rb.build("HiLoFuseNet", 64, 2, dev)), 333825)
        self.assertEqual(_n_params(rb.build("LSTM", 64, 2, dev)), 428289)
        with self.assertRaises(ValueError):
            rb.build("Transformer", 64, 2, dev)

    def test_forward_gives_one_logit_per_window(self):
        torch.manual_seed(0)
        for B in (3, 1):
            x = torch.randn(B, 20, 300, 2)
            for dec in rb.DECODERS:
                m = rb.build(dec, 20, 2, torch.device("cpu")).eval()
                with torch.no_grad():
                    out = m(x)
                self.assertEqual(tuple(out.shape), (B,), dec)
                self.assertTrue(torch.isfinite(out).all(), dec)

    def test_spatial_filters_are_max_norm_constrained(self):
        """HiLoFuseNet's spatial conv renormalises its filters on every forward."""
        m = HiLoFuseNet(C=8, F=2, lstm_hidden=16, D=4, output_size=1).eval()
        conv = m.spatialConv[0]
        with torch.no_grad():
            conv.weight.mul_(10.0)
            m(torch.randn(2, 8, 300, 2))
        norms = conv.weight.detach().reshape(conv.weight.shape[0], -1).norm(dim=1)
        self.assertTrue(bool((norms <= 1.0 + 1e-5).all()))


class TestProtocol(unittest.TestCase):
    """Defaults and data handling that the published numbers depend on."""

    def test_defaults_are_the_paper_run(self):
        a = _args("--decoder", "HiLoFuseNet")
        want = dict(endpoint="sentence_onset", n_folds=4, val_frac=0.15, epochs=40,
                    patience=10, lr=1e-3, weight_decay=1e-4, batch_size=64,
                    hidden=256, dropout=0.5, D=16, seed=42, event_seed=42,
                    win_sec=1.5, pre_sec=0.0, max_per_class=900,
                    neg_mode="upstream", train_mode="pooled", seeds=[42, 1, 2])
        self.assertEqual({k: getattr(a, k) for k in want}, want)
        self.assertEqual(a.subjects, ["sub_1", "sub_2", "sub_3", "sub_4", "sub_6",
                                      "sub_7", "sub_10", "sub_5", "sub_8", "sub_9"])
        self.assertEqual(rb.setting_tags(a), [])

    def test_training_seed_does_not_redraw_events(self):
        """The paper's seeds 1 and 2 score the seed-42 events.

        The extractor may key its cache and draw its events on either `seed` or
        `event_seed`, so both must carry the event seed.
        """
        for seed in (42, 1, 2):
            a = _args("--decoder", "LSTM", "--seed", str(seed), "--endpoint", "word_nonword")
            fa = rb.feature_args(a)
            self.assertEqual((fa.seed, fa.event_seed), (42, 42))
            self.assertEqual(fa.endpoint, "word_nonword")
            self.assertEqual(fa.neg_mode, "upstream")
            self.assertEqual((fa.win_sec, fa.pre_sec), (1.5, 0.0))
        a = _args("--decoder", "LSTM", "--endpoint", "word_nonword",
                  "--neg_mode", "short_silence")
        self.assertEqual(rb.feature_args(a).neg_mode, "short_silence")

    def test_paper_settings_keep_the_paper_file_name(self):
        """paper_cells/btb and scripts/aggregate_btb.py know the files by this name."""
        for ep in rb.ENDPOINTS:
            for seed in (42, 1, 2):
                a = _args("--decoder", "LSTM", "--endpoint", ep, "--seed", str(seed))
                path = rb.result_path("r", a)
                self.assertEqual(os.path.basename(path), f"results_pooled_{ep}_s{seed}.json")
                self.assertTrue((CELLS / "base_LSTM" / os.path.basename(path)).is_file())

    def test_every_nonpaper_setting_gets_its_own_file_name(self):
        """A variant or smoke run can never overwrite, or be summarised as, a paper run."""
        base = ["--decoder", "LSTM", "--endpoint", "word_nonword"]
        variants = {
            "paper": [],
            "event_seed": ["--event_seed", "7"],
            "neg_mode": ["--neg_mode", "short_silence"],
            "per_subject": ["--train_mode", "per_subject"],
            "subset": ["--subjects", "sub_1", "sub_2"],
            "other_subset": ["--subjects", "sub_3", "sub_4"],
            "reordered": ["--subjects"] + list(reversed(rb.SUBS10)),
        }
        nonpaper = {"max_per_class": "60", "n_folds": "2", "val_frac": "0.2",
                    "win_sec": "1.0", "pre_sec": "-0.5", "epochs": "1",
                    "patience": "1", "lr": "3e-4", "weight_decay": "0.01",
                    "batch_size": "32", "hidden": "8", "dropout": "0.1", "D": "4"}
        self.assertEqual(set(nonpaper), set(rb.PAPER))
        for k, v in nonpaper.items():
            variants[k] = [f"--{k}", v]
        names = {k: os.path.basename(rb.result_path("r", _args(*base, *v)))
                 for k, v in variants.items()}
        self.assertEqual(names["paper"], "results_pooled_word_nonword_s42.json")
        self.assertEqual(len(set(names.values())), len(names), names)
        self.assertEqual(names["event_seed"], "results_pooled_word_nonword_s42_ev7.json")
        self.assertEqual(names["subset"], "results_pooled_word_nonword_s42_subj1-2.json")
        self.assertTrue(names["per_subject"].startswith("results_per_subject_"))
        # Per-subject training resets the seed per subject: order cannot matter.
        ps = ["--train_mode", "per_subject", "--subjects"]
        self.assertEqual(rb.result_path("r", _args(*base, *ps, "sub_2", "sub_1")),
                         rb.result_path("r", _args(*base, *ps, "sub_1", "sub_2")))
        self.assertEqual(rb.result_path("r", _args(*base, *ps, *reversed(rb.SUBS10))),
                         rb.result_path("r", _args(*base, "--train_mode", "per_subject")))

    def test_streams_are_stacked_high_then_low(self):
        rs = np.random.RandomState(0)
        lo = rs.randn(4, 3, 192).astype(np.float32)
        hi = rs.randn(4, 3, 300).astype(np.float32)
        X = rb.stack_hi_lo(lo, hi)
        self.assertEqual(X.shape, (4, 3, 300, 2))
        self.assertEqual(X.dtype, np.float32)
        np.testing.assert_array_equal(X[..., 0], hi)
        from scipy.signal import resample_poly
        np.testing.assert_allclose(
            X[..., 1], resample_poly(lo.astype(np.float64), 300, 192, axis=-1), atol=1e-6)
        same = rb.stack_hi_lo(hi, hi)
        np.testing.assert_array_equal(same[..., 1], hi)

    def test_zscore_is_fitted_on_fit_only(self):
        rs = np.random.RandomState(1)
        fit = rs.randn(50, 3, 30, 2).astype(np.float32) * 4 + 2
        fit[:, 1, :, 0] = 5.0                       # a constant channel
        other = rs.randn(20, 3, 30, 2).astype(np.float32) * 100
        zf, zo = rb.zscore_fit_apply(fit, other)
        zf2, _ = rb.zscore_fit_apply(fit, other * 3)
        np.testing.assert_array_equal(zf, zf2)      # val/test cannot move the stats
        np.testing.assert_allclose(zf[:, 0].mean(axis=(0, 1)), 0, atol=1e-5)
        np.testing.assert_allclose(zf[:, 0].std(axis=(0, 1)), 1, atol=1e-4)
        self.assertTrue(np.isfinite(zf).all() and np.isfinite(zo).all())
        np.testing.assert_array_equal(zf[:, 1, :, 0], 0.0)

    def test_event_check_refuses_foreign_features(self):
        """Features from another endpoint or event seed must stop the run."""
        drawn = np.array([1.0, 2.0, 3.0, 4.0]); lab = np.array([1, 0, 1, 0])
        rb.check_events("s", drawn[[0, 2, 3]], lab[[0, 2, 3]], drawn, lab, "x")
        with self.assertRaises(SystemExit):
            rb.check_events("s", np.array([1.0, 2.5]), np.array([1, 0]), drawn, lab, "x")
        with self.assertRaises(SystemExit):
            rb.check_events("s", drawn, 1 - lab, drawn, lab, "x")


def _synthetic(n_subj=2, n=200, T=40, seed=0):
    """Two subjects with different electrode counts and a learnable HIGH-stream signal."""
    rs = np.random.RandomState(seed)
    data, times = {}, {}
    for i in range(n_subj):
        C = 3 + 2 * i
        y = rs.randint(0, 2, n).astype(np.float32)
        X = rs.randn(n, C, T, 2).astype(np.float32)
        X[:, 0, :, 0] += 1.5 * y[:, None]
        data[f"sub_{i}"] = (X, y)
        times[f"sub_{i}"] = np.cumsum(rs.uniform(0.8, 1.2, n))
    return data, times


class TestPooledTraining(unittest.TestCase):
    """The training loop end to end on CPU, without BrainTreebank."""

    def _args(self, dec, **over):
        a = _args("--decoder", dec, "--hidden", "16", "--D", "4", "--batch_size", "32",
                  "--epochs", "3", "--n_folds", "2")
        for k, v in over.items():
            setattr(a, k, v)
        return a

    def test_folds_are_causal_and_reported(self):
        data, times = _synthetic()
        subs = list(data)
        folds, reports, fp = rb.make_folds(subs, data, times, 2, 0.15)
        self.assertAlmostEqual(fp, 40 / 200)        # measured from T, not a literal
        self.assertEqual(len(reports), 2 * len(subs))
        for r in reports:
            self.assertTrue(r["causal"])
            self.assertEqual(r["overlapping_train_test_pairs"], 0)
            self.assertEqual(r["overlapping_val_test_pairs"], 0)
            self.assertEqual(r["embargo_sec"], 7.0)
        for s in subs:
            for fit, va, te in folds[s]:
                t = times[s]
                self.assertLess(t[fit].max(), t[va].min())
                self.assertLess(t[va].max(), t[te].min())
        with self.assertRaises(ValueError):          # early stopping needs val
            rb.make_folds(subs, data, times, 2, 0.0)

    def test_every_decoder_trains_pooled_over_unequal_subjects(self):
        data, times = _synthetic()
        subs = list(data)
        folds, _, _ = rb.make_folds(subs, data, times, 2, 0.15)
        for dec in rb.DECODERS:
            torch.manual_seed(0)
            a = self._args(dec, epochs=1)
            per_fold, record = rb.run_pooled(subs, data, folds, a, torch.device("cpu"),
                                             log=lambda m: None)
            self.assertEqual(sorted(per_fold), sorted(subs), dec)
            for s in subs:
                self.assertEqual(len(per_fold[s]), 2, dec)
                self.assertTrue(all(0.0 <= v <= 1.0 for v in per_fold[s]), dec)
            self.assertEqual([r["fold"] for r in record], [0, 1])

    def test_a_last_batch_of_one_window_trains(self):
        """n_fit % batch_size == 1 leaves one window in the last batch.

        A second squeeze of the (1,) output gave a 0-d tensor that the loss
        rejected against its (1,) target.
        """
        data, times = _synthetic()
        subs = list(data)
        folds, _, _ = rb.make_folds(subs, data, times, 2, 0.15)
        n_fit0 = sum(len(folds[s][0][0]) for s in subs)
        bs = n_fit0 - 1
        self.assertGreater(bs, 1)
        for dec in rb.DECODERS:
            torch.manual_seed(0)
            per_fold, record = rb.run_pooled(subs, data, folds,
                                             self._args(dec, epochs=1, batch_size=bs),
                                             torch.device("cpu"), log=lambda m: None)
            self.assertEqual(record[0]["n_fit"] % bs, 1, dec)
            self.assertTrue(all(np.isfinite(v).all() for v in per_fold.values()), dec)

    def test_learns_a_real_signal(self):
        data, times = _synthetic(n=240)
        subs = list(data)
        folds, _, _ = rb.make_folds(subs, data, times, 2, 0.15)
        torch.manual_seed(0)
        per_fold, record = rb.run_pooled(subs, data, folds, self._args("HiLoFuseNet"),
                                         torch.device("cpu"), log=lambda m: None)
        self.assertGreater(np.mean([np.mean(v) for v in per_fold.values()]), 0.7)
        for r in record:
            self.assertLessEqual(r["best_epoch"], r["epochs_run"])

    def test_patience_stops_training(self):
        data, times = _synthetic()
        subs = list(data)
        folds, _, _ = rb.make_folds(subs, data, times, 2, 0.15)
        torch.manual_seed(0)
        _, record = rb.run_pooled(subs, data, folds,
                                  self._args("LSTM", epochs=30, patience=1, lr=0.0),
                                  torch.device("cpu"), log=lambda m: None)
        # lr=0 never improves on epoch 1, so every fold stops at epoch 2.
        self.assertEqual([r["epochs_run"] for r in record], [2, 2])
        self.assertEqual([r["best_epoch"] for r in record], [1, 1])

    def test_per_subject_training_is_independent_of_the_other_subjects(self):
        """--train_mode per_subject: one model per subject, as wide as that subject."""
        data, times = _synthetic()
        subs = list(data)
        folds, _, _ = rb.make_folds(subs, data, times, 2, 0.15)
        a = self._args("LSTM", epochs=2, train_mode="per_subject", seed=3)
        both, record = rb.train(subs, data, folds, a, torch.device("cpu"), log=lambda m: None)
        self.assertEqual([r["subj"] for r in record], ["sub_0", "sub_0", "sub_1", "sub_1"])
        for s in subs:
            fit_s = len(folds[s][0][0])
            self.assertEqual(next(r for r in record if r["subj"] == s)["n_fit"], fit_s)
        alone, _ = rb.train(["sub_1"], data, folds, a, torch.device("cpu"), log=lambda m: None)
        self.assertEqual(alone["sub_1"], both["sub_1"])


class TestSummary(unittest.TestCase):
    """The published cell: seed mean per subject, then mean ± SD (ddof=1)."""

    SUBS = list(rb.SUBS10)

    def _a(self, *extra):
        return _args("--summarize", "--endpoint", "word_nonword", *extra)

    def _write(self, root, seed, per, a=None, **extra):
        a = a or self._a()
        rec = {"decoder": "HiLoFuseNet", "endpoint": "word_nonword", "seed": seed,
               "event_seed": 42, "per_subject_auroc": per, "folds": 4, "val_frac": 0.15,
               "hp": {"lr": 1e-3, "hidden": 256, "dropout": 0.5, "D": 16,
                      "weight_decay": 1e-4, "epochs": 40, "patience": 10,
                      "batch_size": 64},
               "features": {"win_sec": 1.5, "pre_sec": 0.0, "max_per_class": 900,
                            "neg_mode": "upstream"}}
        for k, v in extra.items():
            if isinstance(v, dict) and isinstance(rec.get(k), dict):
                rec[k] = {**rec[k], **v}
            else:
                rec[k] = v
        with open(rb.result_path(root, a, seed), "w") as fh:
            json.dump(rec, fh)

    def _three(self, root, a=None, **extra_s2):
        rs = np.random.RandomState(0)
        M = rs.uniform(0.5, 0.9, (3, 10))
        for seed, row in zip((42, 1, 2), M):
            self._write(root, seed, dict(zip(self.SUBS, row)), a,
                        **(extra_s2 if seed == 2 else {}))
        return M

    def test_summary_matches_the_table_convention(self):
        with tempfile.TemporaryDirectory() as root:
            M = self._three(root)
            s = rb.summarize(root, self._a())
        per = M.mean(axis=0)
        self.assertAlmostEqual(s["mean"], per.mean(), places=12)
        self.assertAlmostEqual(s["cross_subject_sd"], per.std(ddof=1), places=12)
        self.assertAlmostEqual(s["cross_seed_sd"], M.mean(axis=1).std(ddof=1), places=12)

    def test_summary_refuses_seeds_run_at_other_settings(self):
        """Whatever the file names say, differing recorded settings are refused."""
        bad = {"event_seed": 1, "folds": 2, "val_frac": 0.3, "decoder": "LSTM",
               "hp": {"epochs": 1}, "features": {"max_per_class": 60}}
        for k, v in bad.items():
            with self.subTest(setting=k), tempfile.TemporaryDirectory() as root:
                self._three(root, **{k: v})
                with self.assertRaises(ValueError):
                    rb.summarize(root, self._a())

    def test_summary_refuses_a_file_that_is_not_the_requested_run(self):
        with tempfile.TemporaryDirectory() as root:
            self._three(root)
            with self.assertRaises(ValueError):          # the files are HiLoFuseNet
                rb.summarize(root, self._a("--decoder", "LSTM"))
            with self.assertRaises(FileNotFoundError):   # nothing at --n_folds 2
                rb.summarize(root, self._a("--n_folds", "2"))
            with self.assertRaises(ValueError):          # one seed counted twice
                rb.summarize(root, self._a("--seeds", "42", "42", "1"))
            nine = self._a("--subjects", *self.SUBS[:9])
            self._three(root, nine)                      # ten subjects, nine-subject name
            with self.assertRaises(ValueError):
                rb.summarize(root, nine)

    def test_summary_names_carry_event_seed_and_seeds(self):
        names = {os.path.basename(rb.summary_path("r", self._a(*x))) for x in
                 ([], ["--event_seed", "1"], ["--seeds", "42"], ["--max_per_class", "60"])}
        self.assertEqual(len(names), 4)
        self.assertIn("summary_pooled_word_nonword.json", names)

    def test_paper_cells_give_the_published_table_20_rows(self):
        """--summarize over the paper runs reprints Table 20's scratch rows."""
        table20 = {  # mean, cross-subject SD, cross-seed SD; Task A then Task B
            "HiLoFuseNet": (("0.5330", "0.0220", "0.0217"), ("0.7208", "0.0886", "0.0018")),
            "CNN_LSTM": (("0.5071", "0.0142", "0.0055"), ("0.5706", "0.0469", "0.0032")),
            "LSTM": (("0.5073", "0.0136", "0.0036"), ("0.5618", "0.0429", "0.0006")),
        }
        for dec, cells in table20.items():
            for ep, want in zip(rb.ENDPOINTS, cells):
                a = _args("--summarize", "--decoder", dec, "--endpoint", ep)
                s = rb.summarize(str(CELLS / f"base_{dec}"), a)
                got = tuple(f"{s[k]:.4f}" for k in ("mean", "cross_subject_sd",
                                                    "cross_seed_sd"))
                self.assertEqual(got, want, f"{dec} {ep}")


def _fake_features(subj, n=120, T=300):
    """What extract_subject returns, for a made-up subject: 1.5 s windows 2 s apart."""
    rs = np.random.RandomState(sum(map(ord, subj)))
    C = 3 + len(subj) % 3
    y = np.tile([1, 0], n // 2).astype(np.int64)
    x_hi = rs.randn(n, C, T).astype(np.float32)
    x_hi[:, 0] += 0.8 * y[:, None]
    x_lo = rs.randn(n, C, 192).astype(np.float32)
    ev = 10.0 + 2.0 * np.arange(n)
    return x_lo, x_hi, y, np.zeros((C, 3), np.float32), ev


class TestMain(unittest.TestCase):
    """main() end to end: event check, default folder, file name and fields."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        quiet = mock.patch("sys.stdout", new_callable=io.StringIO)   # the run's log
        quiet.start(); self.addCleanup(quiet.stop)
        env = mock.patch.dict(os.environ, {"CORTEG_OUTPUT_ROOT": self.tmp.name})
        env.start(); self.addCleanup(env.stop)
        import experiments.run_btb_classification as rbc
        ext = mock.patch.object(rbc, "extract_subject",
                                side_effect=lambda s, fa: _fake_features(s))
        ext.start(); self.addCleanup(ext.stop)
        self.shift = 0.0
        exp = mock.patch.object(
            rb, "expected_events",
            side_effect=lambda s, a: (_fake_features(s)[4] + self.shift,
                                      _fake_features(s)[2]))
        exp.start(); self.addCleanup(exp.stop)

    def _root(self, dec):
        return Path(self.tmp.name) / "braintreebank" / f"base_{dec}"

    def test_a_small_run_trains_and_is_named_as_one(self):
        out = rb.main(["--decoder", "LSTM", "--subjects", "sub_a", "sub_b",
                       "--n_folds", "2", "--epochs", "1", "--hidden", "8"])
        self.assertEqual(Path(out), self._root("LSTM") /
                         "results_pooled_sentence_onset_s42_subja-b_f2_ep1_h8.json")
        self.assertFalse((self._root("LSTM") / "results_pooled_sentence_onset_s42.json")
                         .exists())
        with open(out) as fh:
            d = json.load(fh)
        self.assertEqual((d["decoder"], d["endpoint"], d["seed"], d["event_seed"]),
                         ("LSTM", "sentence_onset", 42, 42))
        self.assertEqual(sorted(d["per_subject_auroc"]), ["sub_a", "sub_b"])
        self.assertEqual({s: len(v) for s, v in d["per_subject_fold_auroc"].items()},
                         {"sub_a": 2, "sub_b": 2})
        self.assertEqual((d["folds"], d["hp"]["epochs"], d["features"]["max_per_class"]),
                         (2, 1, 900))
        self.assertEqual(d["nonpaper_settings"], ["subja-b", "f2", "ep1", "h8"])
        self.assertEqual(d["train_mode"], "pooled")
        self.assertEqual(len(d["splits"]), 4)
        self.assertTrue(all(r["overlapping_train_test_pairs"] == 0 for r in d["splits"]))

    def test_features_for_other_events_stop_the_run(self):
        self.shift = 0.5
        with self.assertRaises(SystemExit):
            rb.main(["--decoder", "LSTM", "--subjects", "sub_a", "sub_b",
                     "--n_folds", "2", "--epochs", "1", "--hidden", "8"])
        self.assertFalse(self._root("LSTM").exists())

    def test_bad_requests_are_refused(self):
        for argv in (["--decoder", "LSTM", "--subjects", "sub_a", "sub_a", "sub_b"],
                     ["--decoder", "LSTM", "--neg_mode", "short_silence"],
                     ["--summarize"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit), \
                    mock.patch("sys.stderr"):
                rb.main(argv)

    def test_paper_settings_write_what_the_table_tools_read(self):
        """At the paper settings: the paper file name, the fields the aggregator
        reads, and a --summarize that finds all three seeds by default."""
        rs = np.random.RandomState(0)
        scores = {}

        def fake_train(subs, data, folds, a, dev, log=print):
            self.assertEqual(len(folds[subs[0]]), 4)
            scores[a.seed] = {s: list(rs.uniform(0.5, 0.8, 4)) for s in subs}
            return scores[a.seed], [{"fold": f} for f in range(4)]

        with mock.patch.object(rb, "train", side_effect=fake_train):
            for seed in (42, 1, 2):
                rb.main(["--decoder", "CNN_LSTM", "--endpoint", "word_nonword",
                         "--seed", str(seed)])
        root = self._root("CNN_LSTM")
        self.assertEqual(sorted(os.listdir(root)),
                         [f"results_pooled_word_nonword_s{s}.json" for s in (1, 2, 42)])
        s = rb.main(["--summarize", "--decoder", "CNN_LSTM", "--endpoint", "word_nonword"])
        per = np.array([[np.mean(scores[k][sub]) for sub in rb.SUBS10] for k in (42, 1, 2)])
        self.assertAlmostEqual(s["mean"], per.mean(axis=0).mean(), places=12)
        self.assertTrue((root / "summary_pooled_word_nonword.json").is_file())
        with self.assertRaises(ValueError):
            rb.main(["--summarize", "--decoder", "LSTM", "--endpoint", "word_nonword",
                     "--save_root", str(root)])

        # The same file, read by scripts/aggregate_btb.py, is the published row.
        spec = importlib.util.spec_from_file_location(
            "aggregate_btb", REPO / "scripts" / "aggregate_btb.py")
        agg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agg)
        path = root / "results_pooled_word_nonword_s42.json"
        with open(path) as fh:
            d = json.load(fh)
        self.assertEqual(d["features"]["neg_mode"], "upstream")
        recs = agg.classify(str(path), d)
        self.assertEqual([(r.row, r.task, r.seed) for r in recs],
                         [("cnn_lstm", "word_nonword", 42)])
        self.assertEqual(recs[0].scores, d["per_subject_auroc"])


class TestScript(unittest.TestCase):
    """The script must call the runner with flags it has, at the paper settings."""

    CALL = "python -m experiments.run_btb_baselines"

    def setUp(self):
        self.text = SCRIPT.read_text(encoding="utf-8")
        self.parser = rb.build_parser()
        self.actions = {s: a for a in self.parser._actions for s in a.option_strings}

    def test_every_flag_is_known_and_valid(self):
        self.assertIn(self.CALL, self.text)
        problems = []
        for flag, val in script_flags(self.text):
            act = self.actions.get(flag)
            if act is None:
                problems.append(f"unknown flag {flag}")
            elif act.nargs == 0 and val is not None:
                problems.append(f"{flag} takes no value but got {val!r}")
            elif act.choices and val is not None and val not in act.choices:
                problems.append(f"{flag}={val!r} not in {sorted(act.choices)}")
        self.assertEqual(problems, [], "\n" + "\n".join(problems))

    def test_script_values_equal_the_runner_defaults(self):
        """A literal in the script that disagrees with the default is a second truth."""
        problems = []
        for flag, val in script_flags(self.text):
            act = self.actions.get(flag)
            if act is None or val is None or act.nargs == 0 or act.nargs == "+":
                continue
            conv = act.type or str
            if conv(val) != act.default:
                problems.append(f"{flag} {val} != default {act.default}")
        self.assertEqual(problems, [], "\n" + "\n".join(problems))

    def test_loops_cover_every_decoder_endpoint_and_seed(self):
        loops = dict(re.findall(r"for (\w+) in ([^;]+); do", self.text))
        self.assertEqual(loops, {"DEC": "$DECODERS", "EP": "$ENDPOINTS", "SEED": "$SEEDS"})
        defaults = dict(re.findall(r'^(\w+)="\$\{\1:-(?:\$\{\w+:-)?([^}]*)\}', self.text,
                                   re.M))
        self.assertEqual(defaults["DECODERS"].split(), list(rb.DECODERS))
        self.assertEqual(defaults["ENDPOINTS"].split(), list(rb.ENDPOINTS))
        self.assertEqual(defaults["SEEDS"].split(), ["42", "1", "2"])
        self.assertEqual(set(rb.DECODERS), set(self.actions["--decoder"].choices))
        self.assertEqual(set(rb.ENDPOINTS), set(self.actions["--endpoint"].choices))
        self.assertIn("--event_seed 42", self.text)

    def test_script_runs_from_any_directory(self):
        """`python -m experiments...` needs the repository root as cwd."""
        lines = [ln.strip() for ln in self.text.splitlines()]
        i = lines.index("set -euo pipefail")
        self.assertEqual(lines[i + 1], 'cd "$(dirname "${BASH_SOURCE[0]}")/.."')
        self.assertLess(i + 1, next(k for k, ln in enumerate(lines) if self.CALL in ln))


if __name__ == "__main__":
    unittest.main(verbosity=2)
