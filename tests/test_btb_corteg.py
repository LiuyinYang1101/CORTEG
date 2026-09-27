"""Tests for the BrainTreebank CORTEG runner (experiments/run_btb_classification.py).

They pin what the paper numbers depend on: the two tasks, an event draw that
the training seed cannot change, a cache that refuses features built for
another request or another electrode set, one LoRA adapter per subject that
training actually updates, the training recipe and evaluation cadence of the
paper runs, result names that only the same experiment shares, and a Table 3
script that covers both tasks and all three seeds and writes where the runner
reads its caches.

Everything runs on CPU without the BrainTreebank download or the pretrained
backbone: the data tests build a small synthetic BrainTreebank tree, and the
model tests use a randomly initialised backbone.

Run:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "table3_corteg_braintreebank.sh"
CONFIG = REPO / "configs" / "steegformer_small.json"
sys.path.insert(0, str(REPO))

import experiments.run_btb_classification as btb  # noqa: E402


# ─────────────────────────── a synthetic BrainTreebank ──────────────────────
SUBJ, TRIAL, MOVIE = "sub_1", "trial001", "synthetic-film"
FS = 512.0                       # low enough to keep the fake HDF5 small
DURATION = 1500.0                # seconds of film
ELECTRODES = ["E1", "E2", "E3", "E4"]


def _write_fake_btb(root: Path, popt: Path) -> None:
    """A minimal tree with every file the CORTEG path reads.

    Speech comes in 20 s runs of 0.3 s words every 0.4 s, separated by 10 s of
    silence, so Task A has sentence-initial and mid-sentence words and Task B
    has word-free tiles to draw negatives from.
    """
    rng = np.random.RandomState(0)
    (root / "subject_metadata").mkdir(parents=True)
    (root / "subject_metadata" / f"{SUBJ}_{TRIAL}_metadata.json").write_text(
        json.dumps({"filename": MOVIE}), encoding="utf-8")

    (root / "transcripts" / MOVIE).mkdir(parents=True)
    with open(root / "transcripts" / MOVIE / "features.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["start", "end", "is_onset"])
        t, i = 2.0, 0
        while t < DURATION - 2.0:
            for k in range(50):                       # one 20 s run of speech
                s = t + 0.4 * k
                if s > DURATION - 2.0:
                    break
                w.writerow([f"{s:.3f}", f"{s + 0.3:.3f}", int(i % 8 == 0)])
                i += 1
            t += 30.0

    (root / "subject_timings").mkdir(parents=True)
    with open(root / "subject_timings" / f"{SUBJ}_{TRIAL}_timings.csv", "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["movie_time", "index"])
        for mt in np.arange(0.0, DURATION + 1e-9, 1.0):
            w.writerow([f"{mt:.1f}", f"{mt * FS:.1f}"])

    (root / "electrode_labels" / SUBJ).mkdir(parents=True)
    (root / "electrode_labels" / SUBJ / "electrode_labels.json").write_text(
        json.dumps(ELECTRODES), encoding="utf-8")
    (root / "localization").mkdir(parents=True)
    with open(root / "localization" / "elec_coords_full.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Subject", "Electrode", "Z", "X", "Y"])
        for e in ELECTRODES:
            x, y, z = rng.uniform(-60, 60, 3)
            w.writerow([SUBJ, e, f"{z:.2f}", f"{x:.2f}", f"{y:.2f}"])

    import h5py
    (root / "all_subject_data").mkdir(parents=True)
    with h5py.File(root / "all_subject_data" / f"{SUBJ}_{TRIAL}.h5", "w") as h5:
        for ci in range(len(ELECTRODES)):
            h5.create_dataset(f"data/electrode_{ci}",
                              data=rng.randn(int(DURATION * FS) + 1).astype(np.float32))

    (popt / "electrode_selections").mkdir(parents=True)
    (popt / "electrode_selections" / "clean_laplacian.json").write_text(
        json.dumps({SUBJ: ELECTRODES[:3]}), encoding="utf-8")


def _transcript_rows(root: Path):
    with open(root / "transcripts" / MOVIE / "features.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _event_args(**over):
    a = dict(endpoint="sentence_onset", neg_mode="upstream", event_seed=42,
             max_per_class=60, win_sec=1.5, pre_sec=0.0, hga_low=70.0,
             hga_high=200.0, trial=None, no_cache=False, seed=42)
    a.update(over)
    return argparse.Namespace(**a)


class _FakeBTB(unittest.TestCase):
    """Points BTB_DATA_ROOT, POPT_REPO and the output root at a synthetic tree."""

    @classmethod
    def setUpClass(cls):
        try:
            import h5py  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("h5py not installed")
        cls._tmp = tempfile.TemporaryDirectory()
        base = Path(cls._tmp.name)
        cls.root, cls.popt, cls.out = base / "btb", base / "popt", base / "out"
        _write_fake_btb(cls.root, cls.popt)
        cls._env = mock.patch.dict(os.environ, {
            "BTB_DATA_ROOT": str(cls.root), "POPT_REPO": str(cls.popt),
            "CORTEG_OUTPUT_ROOT": str(cls.out)})
        cls._env.start()

    @classmethod
    def tearDownClass(cls):
        cls._env.stop()
        cls._tmp.cleanup()


# ─────────────────────────────── events ─────────────────────────────────────
class TestEvents(_FakeBTB):

    def test_event_draw_ignores_the_training_seed(self):
        """Seeds 42, 1 and 2 must score one draw; only event_seed changes it."""
        for ep in btb.ENDPOINTS:
            t42, y42, _ = btb.select_events(str(self.root), SUBJ, TRIAL,
                                            _event_args(endpoint=ep, seed=42))
            t1, y1, _ = btb.select_events(str(self.root), SUBJ, TRIAL,
                                          _event_args(endpoint=ep, seed=1))
            np.testing.assert_array_equal(t42, t1, err_msg=ep)
            np.testing.assert_array_equal(y42, y1, err_msg=ep)
            t7, _, _ = btb.select_events(str(self.root), SUBJ, TRIAL,
                                         _event_args(endpoint=ep, event_seed=7))
            self.assertFalse(np.array_equal(t42, t7), f"{ep}: event_seed is ignored")

    def test_task_a_contrasts_words_with_words(self):
        """Task A labels come from is_onset, and both classes are words."""
        t, y, _ = btb.select_events(str(self.root), SUBJ, TRIAL, _event_args())
        rows = _transcript_rows(self.root)
        onset = {round(float(r["start"]), 3): int(r["is_onset"]) for r in rows}
        self.assertTrue(all(round(float(s), 3) in onset for s in t), "a non-word event")
        self.assertEqual([onset[round(float(s), 3)] for s in t], y.astype(int).tolist())
        self.assertEqual(int(y.sum()), 60)
        self.assertEqual(int((1 - y).sum()), 60)
        self.assertTrue(np.all(np.diff(t) >= 0), "events are not in temporal order")

    def test_task_b_contrasts_words_with_silence(self):
        """Positives are word onsets; negatives are centres of word-free 1 s tiles."""
        t, y, meta = btb.select_events(str(self.root), SUBJ, TRIAL,
                                       _event_args(endpoint="word_nonword"))
        rows = _transcript_rows(self.root)
        lo = np.array([float(r["start"]) for r in rows])
        hi = np.array([float(r["end"]) for r in rows])
        pos, neg = t[y == 1], t[y == 0]
        self.assertEqual(len(pos), 60)
        self.assertEqual(len(neg), 60)
        self.assertTrue(np.all(np.isin(np.round(pos, 3), np.round(lo, 3))))
        for c in neg:
            self.assertFalse(np.any((lo < c + 0.5) & (hi > c - 0.5)),
                             f"negative at {c} overlaps a word")
        self.assertEqual(meta["neg_mode"], "upstream")

    def test_task_b_candidates_are_filtered_with_the_5s_window(self):
        """CORTEG reads 1.5 s, but its Task B events are the 5 s arms' events.

        Filtering with CORTEG's own window would admit events near the ends of
        the trigger range that the paper runs and the FM arms never scored.
        """
        t, _, meta = btb.select_events(str(self.root), SUBJ, TRIAL,
                                       _event_args(endpoint="word_nonword", win_sec=1.5))
        self.assertEqual(meta["win_sec"], 5.0)
        t0, t1 = meta["trigger_range"]
        self.assertTrue(np.all(t - 2.5 >= t0) and np.all(t + 2.5 <= t1))

    def test_endpoint_is_required_and_neg_mode_is_task_b_only(self):
        a = _event_args()
        del a.endpoint
        with self.assertRaises(ValueError):
            btb.event_spec(a)
        with self.assertRaises(ValueError):
            btb.event_spec(_event_args(endpoint="sentence_onset", neg_mode="short_silence"))
        self.assertEqual(btb.event_spec(_event_args(endpoint="word_nonword",
                                                    neg_mode="short_silence")),
                         ("word_nonword", "short_silence", 42))


# ─────────────────────────────── cache ──────────────────────────────────────
class TestCacheKey(unittest.TestCase):

    def test_task_a_keeps_the_released_cache_name(self):
        """Existing Task A caches must stay valid; `_s` is now the event seed."""
        a = _event_args(max_per_class=900)
        self.assertEqual(btb.cache_tag("sub_3", "trial000", a),
                         "btb_sub_3_trial000_win1.5_pre0.0_hfa70-200_n900_s42.npz")

    def test_everything_that_selects_events_is_in_the_key(self):
        a = dict(max_per_class=900)
        ref = btb.cache_tag("sub_3", "trial000", _event_args(**a))
        for over in (dict(endpoint="word_nonword"),
                     dict(endpoint="word_nonword", neg_mode="short_silence"),
                     dict(event_seed=1), dict(max_per_class=100),
                     dict(win_sec=5.0), dict(pre_sec=-0.5), dict(hga_high=124.0)):
            self.assertNotEqual(btb.cache_tag("sub_3", "trial000", _event_args(**{**a, **over})),
                                ref, f"{over} does not change the cache key")
        self.assertEqual(btb.cache_tag("sub_3", "trial000", _event_args(**a, seed=2)), ref,
                         "the training seed changes the cache key")
        b = btb.cache_tag("sub_3", "trial000", _event_args(endpoint="word_nonword", **a))
        self.assertIn("word_nonword", b)


class TestExtractSubject(_FakeBTB):
    """The shared entry point: its return contract, its cache and its refusals."""

    def _cache(self, a):
        return self.out / "braintreebank" / "cache" / btb.cache_tag(SUBJ, TRIAL, a)

    def test_contract_and_cache_roundtrip(self):
        for ep in btb.ENDPOINTS:
            a = _event_args(endpoint=ep)
            x_lo, x_hi, y, xyz, ev = btb.extract_subject(SUBJ, a)
            n = len(y)
            self.assertEqual(x_lo.shape, (n, 3, 192))          # 1.5 s at 128 Hz
            self.assertEqual(x_hi.shape, (n, 3, 300))          # 1.5 s at 200 Hz
            self.assertEqual((x_lo.dtype, x_hi.dtype), (np.float32, np.float32))
            self.assertEqual(y.dtype, np.int64)
            self.assertEqual(xyz.shape, (3, 3))
            self.assertLess(float(np.abs(xyz).max()), 1.0, "coordinates are not in metres")
            self.assertEqual(ev.dtype, np.float64)
            self.assertTrue(np.all(np.diff(ev) >= 0))

            # The scored events are the drawn events that fit the recording.
            t, lab, _ = btb.select_events(str(self.root), SUBJ, TRIAL, a)
            keep = np.isin(t, ev)
            np.testing.assert_array_equal(t[keep], ev)
            np.testing.assert_array_equal(lab[keep].astype(np.int64), y)

            with np.load(self._cache(a)) as d:
                self.assertEqual(str(d["endpoint"]), ep)
                self.assertEqual(int(d["event_seed"]), 42)
                self.assertEqual(int(d["max_per_class"]), 60)

            # A second call, and a call with another TRAINING seed, read the cache.
            with mock.patch.object(btb, "extract_windows",
                                   side_effect=AssertionError("cache not used")):
                again = btb.extract_subject(SUBJ, _event_args(endpoint=ep, seed=2))
            for p, q in zip((x_lo, x_hi, y, xyz, ev), again):
                np.testing.assert_array_equal(p, q)

    def test_chunked_transform_is_exact(self):
        """event_chunk bounds memory only; the features must not change."""
        a = _event_args(endpoint="word_nonword", max_per_class=55, no_cache=True)
        whole = btb.extract_subject(SUBJ, argparse.Namespace(**{**vars(a), "event_chunk": 0}))
        chunked = btb.extract_subject(SUBJ, argparse.Namespace(**{**vars(a), "event_chunk": 7}))
        for p, q in zip(whole, chunked):
            np.testing.assert_array_equal(p, q)

    def _plant(self, a, **arrays):
        """A cache for request `a`, legacy-shaped unless `arrays` add fields.

        Its coordinates are the selected electrodes' own, so only what
        `arrays` changes can make it wrong.
        """
        path = self._cache(a)
        path.parent.mkdir(parents=True, exist_ok=True)
        xyz = btb.electrode_selection(str(self.root), SUBJ)[2]
        base = dict(x_lo=np.zeros((4, 3, 192), np.float32),
                    x_hi=np.zeros((4, 3, 300), np.float32),
                    y=np.array([0, 1, 0, 1]), xyz=xyz,
                    event_times=np.arange(4, dtype=np.float64),
                    win_sec=np.float64(1.5), pre_sec=np.float64(0.0))
        base.update(arrays)
        np.savez(path, **{k: v for k, v in base.items() if v is not None})
        return path

    def test_legacy_task_a_cache_is_reused(self):
        """Caches written before the endpoint switch hold Task A events."""
        a = _event_args(max_per_class=7)
        path = self._plant(a)
        try:
            out = btb.extract_subject(SUBJ, a)
            self.assertEqual(out[0].shape, (4, 3, 192))
        finally:
            path.unlink()

    def test_mismatched_caches_are_refused(self):
        cases = [
            # A Task A name holding Task B events.
            (_event_args(max_per_class=8), dict(endpoint=np.array("word_nonword"))),
            # A Task B name with no endpoint recorded: it predates the switch.
            (_event_args(endpoint="word_nonword", max_per_class=8), {}),
            # Another negative pool, another event seed, another window.
            (_event_args(endpoint="word_nonword", max_per_class=8),
             dict(endpoint=np.array("word_nonword"), neg_mode=np.array("short_silence"))),
            (_event_args(max_per_class=8), dict(event_seed=np.int64(1))),
            (_event_args(max_per_class=8), dict(win_sec=np.float64(2.0))),
        ]
        for a, arrays in cases:
            path = self._plant(a, **arrays)
            try:
                with self.assertRaises(RuntimeError, msg=str(arrays)):
                    btb.extract_subject(SUBJ, a)
            finally:
                path.unlink()

    def test_caches_over_another_electrode_set_are_refused(self):
        """Another $POPT_REPO selection changes the features, not the cache name.

        Caches with electrode names are compared by name and order; legacy
        caches, which hold only coordinates, through those.
        """
        a = _event_args(max_per_class=9)
        use, _, xyz = btb.electrode_selection(str(self.root), SUBJ)
        self.assertEqual(use, ELECTRODES[:3])
        cases = {
            "another set": dict(electrodes=np.array(["E1", "E2", "E4"])),
            "another order": dict(electrodes=np.array(["E2", "E1", "E3"])),
            "legacy, other coordinates": dict(xyz=xyz[[1, 0, 2]]),
            "legacy, fewer electrodes": dict(xyz=xyz[:2]),
            "features over another channel count": dict(
                electrodes=np.array(use), x_lo=np.zeros((4, 2, 192), np.float32)),
        }
        for what, arrays in cases.items():
            path = self._plant(a, **arrays)
            try:
                with self.assertRaisesRegex(RuntimeError, "electrode", msg=what):
                    btb.extract_subject(SUBJ, a)
            finally:
                path.unlink()
        # The same selection, recorded by name, is reused.
        path = self._plant(a, electrodes=np.array(use))
        try:
            self.assertEqual(btb.extract_subject(SUBJ, a)[0].shape, (4, 3, 192))
        finally:
            path.unlink()

    def test_cache_is_written_through_a_per_process_temporary(self):
        """Two processes building one cache must not share a temporary file."""
        a = _event_args(max_per_class=11, no_cache=True)
        with mock.patch.object(np, "savez_compressed", wraps=np.savez_compressed) as save:
            btb.extract_subject(SUBJ, a)
        tmp = str(save.call_args[0][0])
        self.assertTrue(tmp.endswith(f".{os.getpid()}.tmp.npz"), tmp)
        self.assertTrue(self._cache(a).exists())
        self.assertEqual(list(self._cache(a).parent.glob("*.tmp.npz")), [])


# ─────────────────────────────── model ──────────────────────────────────────
def _model_args(*extra):
    """A random-init run given no config: main()'s resolution picks the config."""
    args = btb.build_parser().parse_args(["--no_pretrained", *extra])
    with contextlib.redirect_stdout(io.StringIO()):
        btb.resolve_backbone_config(args)
    return args


class _OneTorchThread(unittest.TestCase):
    """Runs its tests with one torch thread, restoring the count afterwards.

    The models here see batches of a few windows over a few channels: one
    thread is faster than many, and on a shared machine, where the threads
    contend, about ten times faster.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import torch
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls._threads)
        super().tearDownClass()


class TestPerSubjectLoRA(_OneTorchThread):
    """The paper's CORTEG row is 'pooled, per-subject LoRA'."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import torch
        cls.torch = torch
        torch.manual_seed(0)
        cls.C = 6
        cls.xyz = np.random.RandomState(0).uniform(-0.06, 0.06, (cls.C, 3)).astype(np.float32)
        cls.model = btb.build_corteg(cls.C, 192, cls.xyz, _model_args())
        cls.model.eval()
        rs = np.random.RandomState(1)
        cls.x = (torch.from_numpy(rs.randn(3, cls.C, 192).astype(np.float32)),
                 torch.from_numpy(np.abs(rs.randn(3, cls.C, 300)).astype(np.float32)),
                 torch.from_numpy(cls.xyz)[None].expand(3, -1, -1))
        with torch.no_grad():
            cls.before = cls.model(cls.x[0], x_hi=cls.x[1], ecog_xyz=cls.x[2])
        cls.n = btb.add_per_subject_lora(cls.model, 10, 4, 0.2)
        cls.model.eval()                 # the new adapters are built in train mode

    def _out(self):
        with self.torch.no_grad():
            return self.model(self.x[0], x_hi=self.x[1], ecog_xyz=self.x[2])

    def test_one_adapter_per_subject_in_the_last_four_blocks(self):
        from models.steegformer.lora import LoRALinear, TaskLoRALinear
        self.assertEqual(self.n, 16, "want 4 blocks x qkv/proj/fc1/fc2")
        blocks = self.model.backbone.blocks
        for bi, blk in enumerate(blocks):
            tl = [m for m in blk.modules() if isinstance(m, TaskLoRALinear)]
            self.assertEqual(len(tl), 4 if bi >= len(blocks) - 4 else 0, f"block {bi}")
            self.assertTrue(all(m.n_tasks == 10 for m in tl))
        self.assertFalse(any(isinstance(m, LoRALinear) for m in self.model.modules()),
                         "a shared LoRA adapter is left over")

    def test_adapters_and_readout_are_trainable(self):
        from models.steegformer.lora import TaskLoRALinear
        trainable = {id(p) for p in self.model.parameters() if p.requires_grad}
        for m in self.model.modules():
            if isinstance(m, TaskLoRALinear):
                for p in list(m.A.parameters()) + list(m.B.parameters()):
                    self.assertIn(id(p), trainable)
                self.assertFalse(m.base.weight.requires_grad, "the base weight is unfrozen")
        self.assertIn(id(self.model.head.head.weight), trainable)

    def test_swap_is_output_preserving_and_the_adapter_follows_the_subject(self):
        from models.steegformer.lora import TaskLoRALinear, set_active_task
        set_active_task(self.model, 0)
        self.torch.testing.assert_close(self._out(), self.before)   # B starts at zero
        layer = next(m for m in self.model.modules() if isinstance(m, TaskLoRALinear))
        with self.torch.no_grad():
            layer.B[3].weight.normal_(0, 0.5)
        try:
            set_active_task(self.model, 3)
            self.assertFalse(self.torch.allclose(self._out(), self.before),
                             "subject 3's adapter is not used")
            set_active_task(self.model, 0)
            self.torch.testing.assert_close(self._out(), self.before)
        finally:
            with self.torch.no_grad():
                layer.B[3].weight.zero_()

    def test_random_init_keeps_the_backbone_config(self):
        """The paper's control trains with drop_path_rate 0.1, like CORTEG.

        The model under test was built from `--no_pretrained` alone, with no
        --model_kwargs_json, through resolve_backbone_config as main() does: the
        fallback must pick the variant's config rather than leave drop_path 0.0.
        """
        args = _model_args()
        self.assertEqual(Path(args.model_kwargs_json).resolve(), CONFIG.resolve())
        dp = self.model.backbone.blocks[-1].drop_path1
        self.assertAlmostEqual(float(getattr(dp, "drop_prob", 0.0)), 0.1, places=6)

    def test_no_config_without_random_init_stops(self):
        """A CORTEG run with no config must stop, not train a random backbone."""
        args = btb.build_parser().parse_args([])
        with self.assertRaises(SystemExit):
            btb.resolve_backbone_config(args)
        given = btb.build_parser().parse_args(["--model_kwargs_json", "x.json"])
        self.assertEqual(btb.resolve_backbone_config(given), "x.json")


# ─────────────────────────────── training ───────────────────────────────────
class TestPooledTraining(_OneTorchThread):
    """A two-subject fold on CPU: runs, and repeats exactly at a fixed seed."""

    @staticmethod
    def _bundles():
        rs = np.random.RandomState(0)
        out = []
        for C in (4, 6):
            n = 80
            t = np.sort(rs.uniform(0, 3000, n))
            y = rs.permutation(np.repeat([0, 1], n // 2)).astype(np.int64)
            x_lo = rs.randn(n, C, 192).astype(np.float32)
            x_hi = np.abs(rs.randn(n, C, 300)).astype(np.float32)
            x_lo[y == 1] += 0.3                              # something to learn
            xyz = rs.uniform(-0.06, 0.06, (C, 3)).astype(np.float32)
            out.append((x_lo, x_hi, y, xyz, t))
        return out

    def _fold(self, bundles):
        from data.braintreebank import forward_chaining_split
        return [forward_chaining_split(b[4], win_sec=1.5, n_folds=2, val_frac=0.15)[0]
                for b in bundles]

    def test_fold_runs_and_is_repeatable(self):
        """Also: training reaches each subject's own adapter and the readout.

        A step that forgot set_active_task, or an optimizer built before the
        per-subject swap, still runs and still repeats exactly; only the
        trained weights show it.
        """
        import torch
        from models.steegformer.lora import TaskLoRALinear
        from train.earlystop import EarlyStopper
        bundles = self._bundles()
        fold = self._fold(bundles)
        args = _model_args("--epochs", "2", "--eval_every", "1", "--batch_size", "8",
                           "--eval_batch_size", "16", "--seed", "3")
        args.warmup_epochs = btb.warmup_epochs_of(args)

        seen, build, restore = {}, btb.build_corteg, EarlyStopper.restore

        def build_and_keep_head(*a, **k):
            m = build(*a, **k)
            seen["head0"] = m.head.head.weight.detach().clone()
            return m

        def restore_and_keep_model(self_, model):
            restore(self_, model)
            seen["model"] = model

        with mock.patch.object(btb, "build_corteg", build_and_keep_head), \
                mock.patch.object(EarlyStopper, "restore", restore_and_keep_model):
            first = btb.run_pooled_fold(bundles, fold, args, "cpu")
        self.assertEqual(sorted(first), [0, 1])
        self.assertTrue(all(0.0 <= v <= 1.0 for v in first.values()))

        layers = [m for m in seen["model"].modules() if isinstance(m, TaskLoRALinear)]
        self.assertEqual(len(layers), 16)
        for i, m in enumerate(layers):
            b0, b1 = m.B[0].weight.detach(), m.B[1].weight.detach()
            self.assertGreater(float(b0.abs().sum()), 0.0, f"layer {i}: subject 0 untrained")
            self.assertGreater(float(b1.abs().sum()), 0.0, f"layer {i}: subject 1 untrained")
            self.assertFalse(torch.allclose(b0, b1), f"layer {i}: one adapter for both")
        self.assertFalse(torch.equal(seen["model"].head.head.weight.detach(), seen["head0"]),
                         "the readout head was not trained")

        # set_seed at the top of every fold seeds Python's `random` too, which
        # drives the sampler's shuffle; without it this second run differs.
        second = btb.run_pooled_fold(bundles, fold, args, "cpu")
        self.assertEqual(first, second)

    def _cadence(self, *flags):
        """'T' per training epoch and 'E' per evaluation, in order.

        Training is stubbed out, so the model never improves after the first
        evaluation, and each later evaluation counts against --patience.
        """
        import warnings
        from train.earlystop import EarlyStopper
        bundles = self._bundles()
        args = _model_args("--eval_batch_size", "64", *flags)
        args.warmup_epochs = btb.warmup_epochs_of(args)
        log, step = [], EarlyStopper.step

        def train(*a, **k):
            log.append("T")
            return {"loss": 0.5}

        def evaluate(self_, score, model):
            log.append("E")
            return step(self_, score, model)

        with mock.patch("train.engine.train_one_epoch", train), \
                mock.patch.object(EarlyStopper, "step", evaluate), warnings.catch_warnings():
            warnings.simplefilter("ignore")       # scheduler stepped with no optimizer step
            btb.run_pooled_fold(bundles, self._fold(bundles), args, "cpu")
        return "".join(log)

    def test_evaluation_cadence_and_patience_in_evaluations(self):
        # Every --eval_every epochs, and always on the last epoch.
        self.assertEqual(self._cadence("--epochs", "3", "--eval_every", "2"), "TTETE")
        # Patience counts evaluations: with eval_every 2, patience 2 stops after
        # the 2nd evaluation without improvement, i.e. 6 epochs, not 4.
        self.assertEqual(self._cadence("--epochs", "20", "--eval_every", "2",
                                       "--patience", "2"), "TTETTETTE")


# ─────────────────────────────── CLI ────────────────────────────────────────
class TestRecipe(unittest.TestCase):
    """The defaults are the recipe recorded in every paper run's config."""

    def test_defaults_are_the_paper_recipe(self):
        a = btb.build_parser().parse_args([])
        want = dict(endpoint="sentence_onset", neg_mode="upstream", event_seed=42,
                    max_per_class=900, win_sec=1.5, pre_sec=0.0, hga_low=70.0,
                    hga_high=200.0, train_mode="pooled", n_folds=4, val_frac=0.15,
                    merge_strategy="layerwise_gate", lora_last_n=4, lora_r=4,
                    lora_alpha=16, lora_dropout=0.2, head_dropout=0.0,
                    per_subject_lora=True, epochs=60, batch_size=16,
                    eval_batch_size=32, accum_iter=4, use_amp=True, lr=3e-4,
                    weight_decay=5e-3, min_lr=1e-5, max_norm=1.0, eval_every=2,
                    patience=15, select_metric="pooled", seed=42)
        got = {k: getattr(a, k) for k in want}
        self.assertEqual(got, want)
        self.assertEqual(a.subjects, ["sub_1", "sub_2", "sub_3", "sub_4", "sub_6",
                                      "sub_7", "sub_10", "sub_5", "sub_8", "sub_9"])
        self.assertEqual(btb.warmup_epochs_of(a), 6)

    def test_opt_outs(self):
        a = btb.build_parser().parse_args(["--shared_lora", "--no_amp",
                                           "--warmup_epochs", "3"])
        self.assertFalse(a.per_subject_lora)
        self.assertFalse(a.use_amp)
        self.assertEqual(btb.warmup_epochs_of(a), 3)

    def test_result_names_do_not_collide(self):
        p = btb.build_parser()
        names = set()
        for ep in btb.ENDPOINTS:
            for seed in ("42", "1", "2"):
                for arm in ([], ["--merge_strategy", "average"], ["--no_pretrained"]):
                    names.add(btb.result_filename(
                        p.parse_args(["--endpoint", ep, "--seed", seed, *arm])))
        self.assertEqual(len(names), 18)
        self.assertEqual(btb.result_filename(p.parse_args([])),
                         "btb_pooled_layerwise_gate_sentence_onset_seed42.json")


class TestResultNames(unittest.TestCase):
    """Two runs share a results file name only when they are the same experiment.

    A shared name means the second run silently overwrites the first, and a
    run at another setting under the paper name is read as the paper cell.
    """

    @staticmethod
    def _departures():
        """{dest: argv} moving one option off its default, for every option."""
        p = btb.build_parser()
        out = {}
        for act in p._actions:
            if act.dest == "help":
                continue
            default, opt = p.get_default(act.dest), act.option_strings[0]
            if act.nargs == 0:                           # a store_true/false pair
                if act.const != default:
                    out[act.dest] = [opt]
                continue
            if act.choices:
                argv = [opt, next(c for c in act.choices if c != default)]
            elif act.dest == "subjects":
                argv = [opt, *reversed(default)]
            elif act.dest == "trial":
                argv = [opt, "trial999"]
            elif act.dest == "model_kwargs_json":
                argv = [opt, json.dumps({"backbone_kwargs": {"drop_path_rate": 0.2}})]
            elif act.dest == "save_root":
                argv = [opt, "/elsewhere"]
            elif act.dest == "warmup_epochs":
                argv = [opt, "3"]                        # the default stands for 6
            elif act.type is int:
                argv = [opt, str(default + 1)]
            elif act.type is float:
                argv = [opt, repr(default * 2 + 0.1)]
            else:
                raise AssertionError(f"no departure defined for --{act.dest}")
            out[act.dest] = argv
        return out

    def test_every_option_is_named_or_deliberately_not(self):
        dests = {a.dest for a in btb.build_parser()._actions if a.dest != "help"}
        groups = [set(btb.NAME_FIELDS), set(btb.NOT_IN_NAME), set(btb.NAMED_SPECIALLY),
                  set(btb.VALUE_TAGS)]
        self.assertEqual(sum(len(g) for g in groups), len(set().union(*groups)),
                         "an option is in two naming groups")
        self.assertEqual(set().union(*groups), dests,
                         "decide how a new option enters the results file name")

    def test_every_departure_changes_the_name(self):
        p = btb.build_parser()
        base = btb.result_filename(p.parse_args([]))
        names = {}
        for dest, argv in self._departures().items():
            name = btb.result_filename(p.parse_args(argv))
            if dest in btb.NOT_IN_NAME:
                self.assertEqual(name, base, f"--{dest} cannot change the numbers")
                continue
            self.assertNotEqual(name, base, f"{argv} writes the paper-setting file")
            self.assertNotIn(name, names.values(), f"{argv} collides with another setting")
            names[dest] = name

    def test_subjects_are_named_by_set_and_pooled_order(self):
        p = btb.build_parser()
        name = lambda *a: btb.result_filename(p.parse_args(list(a)))
        self.assertNotEqual(name("--subjects", "sub_3"), name("--subjects", "sub_9"))
        self.assertIn("_subj3-9_", name("--subjects", "sub_3", "sub_9"))
        numeric = [f"sub_{i}" for i in range(1, 11)]
        self.assertIn("_subj1-2-3-4-5-6-7-8-9-10_", name("--subjects", *numeric))
        self.assertEqual(name("--subjects", *btb.PAPER_SUBJECT_ORDER), name())
        # One model per subject: the order is not part of the run, the set is.
        per = ["--train_mode", "per_subject"]
        self.assertEqual(name(*per, "--subjects", *numeric), name(*per))
        self.assertEqual(name(*per, "--subjects", "sub_9", "sub_3"),
                         name(*per, "--subjects", "sub_3", "sub_9"))
        self.assertNotEqual(name(*per, "--shared_lora"), name(*per))

    def test_the_shipped_config_is_not_a_departure(self):
        p = btb.build_parser()
        self.assertEqual(btb.result_filename(p.parse_args(["--model_kwargs_json", str(CONFIG)])),
                         btb.result_filename(p.parse_args([])))
        self.assertIn("_cfg-steegformer_base_", btb.result_filename(p.parse_args(
            ["--model_kwargs_json", str(REPO / "configs" / "steegformer_base.json")])))
        a = p.parse_args(["--warmup_epochs", "6"])        # what the default stands for
        self.assertEqual(btb.result_filename(a), btb.result_filename(p.parse_args([])))


class TestMain(_FakeBTB):
    """main() end to end on the synthetic tree, with the training stubbed."""

    def _main(self, *argv):
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            btb.main(list(argv))

    def test_settings_that_cannot_run_stop_before_any_work(self):
        for bad in (["--val_frac", "0"], ["--val_frac", "1"], ["--epochs", "0"],
                    ["--n_folds", "0"], ["--eval_every", "0"],
                    ["--subjects", "sub_1", "sub_1"],
                    ["--neg_mode", "short_silence"]):
            with mock.patch.object(btb, "extract_subject",
                                   side_effect=AssertionError("ran")) as ext:
                with self.assertRaises(SystemExit, msg=str(bad)):
                    self._main("--no_pretrained", *bad)
                ext.assert_not_called()

    def test_result_file(self):
        """Fold AUROCs keep NaN; the paper-run keys list finite folds only."""
        folds = iter([{0: float("nan")}, {0: 0.6}])
        with mock.patch.object(btb, "run_pooled_fold", side_effect=lambda *a: next(folds)):
            self._main("--subjects", SUBJ, "--max_per_class", "60", "--n_folds", "2",
                       "--epochs", "1", "--no_pretrained")
        path = (self.out / "braintreebank" / "runs" /
                "btb_pooled_layerwise_gate_sentence_onset_randinit_n60_subj1_f2_ep1_seed42.json")
        self.assertTrue(path.exists(), sorted(p.name for p in path.parent.glob("*")))
        d = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(np.isnan(d["per_subject"][SUBJ]["folds"][0]))
        self.assertEqual(d["per_subject"][SUBJ]["folds"][1], 0.6)
        self.assertEqual(d["per_subject_per_fold"], {SUBJ: [0.6]})
        self.assertEqual(d["per_subject_auroc"], {SUBJ: 0.6})
        self.assertEqual(d["cohort_mean_auroc"], 0.6)
        self.assertEqual(len(d["splits"]), 2)
        self.assertEqual(Path(d["args"]["model_kwargs_json"]).resolve(), CONFIG.resolve())


class TestTable3Script(unittest.TestCase):
    """The script must run what the paper reports: 3 arms x 2 tasks x 3 seeds."""

    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def _default(self, var):
        m = re.search(r'^%s="\$\{%s:-(.*?)\}"' % (var, var), self.text, re.M)
        self.assertIsNotNone(m, f"{var} has no default")
        return m.group(1)

    def test_loops_cover_the_paper(self):
        seeds = self._default("SEEDS")                 # ${SEED:-42 1 2}: SEED still works
        self.assertEqual(seeds.replace("${SEED:-", "").rstrip("}").split(),
                         ["42", "1", "2"])
        self.assertEqual(self._default("ENDPOINTS").split(),
                         ["sentence_onset", "word_nonword"])
        self.assertEqual(self._default("ARMS").split(), ["gate", "average", "randinit"])

    def test_every_call_carries_the_task_seed_and_recipe(self):
        sys.path.insert(0, str(REPO / "tests"))
        from test_release_integrity import _invocations, BTB_RUNNER_CALL
        calls = [c for _, c in _invocations(self.text, BTB_RUNNER_CALL)]
        self.assertEqual(len(calls), 3)
        for c in calls:
            for flag in ('--endpoint "$EP"', '--seed "$SEED"', "--event_seed 42",
                         "--model_kwargs_json", "--per_subject_lora", "--use_amp",
                         "--epochs 60", "--patience 15", "--eval_every 2",
                         "--batch_size 16", "--accum_iter 4", "--lr 3e-4",
                         "--min_lr 1e-5", "--head_dropout 0.0"):
                self.assertIn(flag, c)
        gate, average, randinit = calls
        self.assertIn("--merge_strategy layerwise_gate", gate)
        self.assertIn("--merge_strategy average", average)
        # The control is the gate arm with the checkpoint skipped, nothing else.
        strip = lambda c: set(re.findall(r"--\w+(?: [^-\s][^\s]*)?", c))
        self.assertEqual(strip(randinit) - strip(gate), {"--no_pretrained"})
        self.assertNotIn("--no_pretrained", gate + average)


@unittest.skipUnless(shutil.which("bash"), "needs bash")
class TestTable3ScriptRuns(unittest.TestCase):
    """The script run for real, with a `python` on PATH that only records its argv."""

    def _run(self, *args, **env):
        """(returncode, stdout, [argv of each runner call])."""
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "calls.jsonl"
            shim = Path(d) / "python"
            shim.write_text(f"#!{sys.executable}\n"
                            "import json, sys\n"
                            f"open({str(log)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n",
                            encoding="utf-8")
            shim.chmod(0o755)
            full = {k: v for k, v in os.environ.items()
                    if k not in ("CORTEG_OUTPUT_ROOT", "ECOG_OUTPUT_ROOT", "SEED",
                                 "SEEDS", "ENDPOINTS", "ARMS")}
            full.update(env, PATH=f"{d}{os.pathsep}{os.environ.get('PATH', '')}")
            r = subprocess.run(["bash", str(SCRIPT), *args], env=full, text=True,
                               capture_output=True, timeout=120)
            calls = [json.loads(l) for l in log.read_text().splitlines()] \
                if log.exists() else []
        return r.returncode, r.stdout, calls

    def test_arguments_print_the_header_and_run_nothing(self):
        """`--help` or a typo must not start the 18-run sweep."""
        rc, out, calls = self._run("--help")
        self.assertEqual((rc, calls), (0, []))
        self.assertIn("Tables 3, 9 and 20", out)
        rc, _, calls = self._run("--seeds", "42")
        self.assertEqual((rc, calls), (2, []))

    def test_bad_choices_stop_before_the_first_run(self):
        for env in (dict(ARMS="gate foo"), dict(ENDPOINTS="word_nonword task_c"),
                    dict(SEEDS="42 x")):
            rc, _, calls = self._run(**env)
            self.assertEqual((rc, calls), (2, []), env)

    def test_results_go_where_the_runner_puts_its_caches(self):
        """The script's root must follow paths.get_output_root(), fallbacks included."""
        import paths
        one = dict(ARMS="gate", ENDPOINTS="word_nonword", SEEDS="1")
        for roots in (dict(CORTEG_OUTPUT_ROOT="/c/root"), dict(ECOG_OUTPUT_ROOT="/e/root"),
                      dict(CORTEG_OUTPUT_ROOT="/c/root", ECOG_OUTPUT_ROOT="/e/root"), {}):
            rc, _, calls = self._run(**one, **roots)
            self.assertEqual((rc, len(calls)), (0, 1), roots)
            argv = calls[0]
            got = argv[argv.index("--save_root") + 1]
            env = {k: v for k, v in os.environ.items()
                   if k not in ("CORTEG_OUTPUT_ROOT", "ECOG_OUTPUT_ROOT")}
            with mock.patch.dict(os.environ, {**env, **roots}, clear=True):
                want = os.path.join(paths.get_output_root(), "braintreebank", "table3")
            self.assertEqual(got, want, roots)

    def test_the_sweep_writes_exactly_the_paper_result_names(self):
        """18 calls, and the flags the script spells out add no tag to any name."""
        rc, _, calls = self._run(CORTEG_OUTPUT_ROOT="/c/root")
        self.assertEqual((rc, len(calls)), (0, 18))
        want = {f"btb_pooled_{merge}_{ep}{tag}_seed{seed}.json"
                for ep in btb.ENDPOINTS for seed in (42, 1, 2)
                for merge, tag in (("layerwise_gate", ""), ("average", ""),
                                   ("layerwise_gate", "_randinit"))}
        cwd = os.getcwd()
        os.chdir(REPO)                         # the script runs from the repository root
        try:
            got = set()
            for argv in calls:
                self.assertEqual(argv[:2], ["-m", "experiments.run_btb_classification"])
                a = btb.build_parser().parse_args(argv[2:])
                with contextlib.redirect_stdout(io.StringIO()):
                    btb.resolve_backbone_config(a)
                a.warmup_epochs = btb.warmup_epochs_of(a)
                got.add(btb.result_filename(a))
        finally:
            os.chdir(cwd)
        self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main(verbosity=2)
