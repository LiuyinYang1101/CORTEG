"""Tests for the Stanford intracranial-FM regression runners.

Everything here runs on CPU in seconds and needs neither the dataset nor the
third-party models: the model-dependent steps are exercised with stand-ins.
What is pinned down:

  * the runners' defaults are the settings the paper's numbers were produced
    with, and every call in the shell script parses under the real parser;
  * a native window ends exactly at its target and pairs with the right one;
  * Brant's anchor grid never gives a window context from after its target;
  * the cache names are the ones the paper runs wrote, so their caches stay
    reusable, every non-paper setting and debug cap changes the name, and a
    cached array with the wrong number of windows is refused;
  * only a run with every recorded setting, all nine subjects, no cap and
    the paper's device class (a GPU with AMP) writes into a paper folder; a
    skipped cell and a reused LOO stage 1 must come from exactly the same run;
  * the pooled probe's optional reseed makes it independent of the RNG state
    that extraction leaves behind;
  * what the shell script runs, where it writes, which arguments it takes,
    and that it runs from any directory and on any subject count;
  * the native-file builder writes where the runners look;
  * the timing of BrainBERT's centre-frame pooling, stated in the runner.

Run:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import itertools
import json
import os
import pickle
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "table1_ieeg_fm_stanford.sh"
sys.path.insert(0, str(REPO))


def _script_calls(text: str, fn: str):
    """Token lists of every `fn ...` call in a shell script, continuations joined.

    A call that uses loop variables is expanded over every value its `for`
    loops take; any other variable becomes a placeholder path.
    """
    loops = {}
    for m in re.finditer(r"for (\w+) in ([^;]+);", text):
        loops.setdefault(m.group(1), set()).update(m.group(2).split())
    calls = []
    for line in text.replace("\\\n", " ").splitlines():
        s = line.strip()
        if not s.startswith(fn + " "):
            continue
        used = [v for v in sorted(loops) if re.search(r"\$\{?%s\b" % v, s)]
        for combo in itertools.product(*[sorted(loops[v]) for v in used]):
            env = dict(zip(used, combo))
            s2 = re.sub(r"\$\{(\w+)\}|\$(\w+)",
                        lambda m: env.get(m.group(1) or m.group(2), "/x"), s)
            calls.append(shlex.split(s2)[1:])
    return calls


class TestRecordedDefaults(unittest.TestCase):
    """Defaults must reproduce the paper's cells without extra flags."""

    def _reg(self, *argv):
        from experiments.run_ieeg_fm_regression import build_argparser, resolve_budget
        return resolve_budget(build_argparser().parse_args(["--fm", "popt", *argv]))

    def test_probe(self):
        a = self._reg()
        self.assertEqual((a.mode, a.head, a.epochs, a.early_stop_patience, a.warmup_epochs),
                         ("probe", "linear", 60, 20, 10))
        self.assertEqual((a.lr, a.weight_decay, a.min_lr, a.batch_size, a.val_ratio),
                         (1e-3, 1e-4, 1e-6, 256, 0.1))
        self.assertEqual((a.reref, a.bb_pool), ("laplacian_xyz", "center10"))
        self.assertTrue(a.use_amp)

    def test_finetune(self):
        a = self._reg("--mode", "finetune")
        self.assertEqual((a.epochs, a.early_stop_patience, a.warmup_epochs), (60, 20, 5))
        self.assertEqual((a.unfreeze_last_n, a.ft_lr), (2, 1e-4))

    def test_temporal(self):
        a = self._reg("--head", "temporal")
        self.assertEqual((a.epochs, a.early_stop_patience, a.warmup_epochs), (100, 20, 10))
        self.assertEqual((a.seq_len, a.temporal_hidden, a.temporal_direction), (64, 128, "bi"))

    def test_explicit_flags_win(self):
        a = self._reg("--epochs", "3", "--warmup_epochs", "1", "--no_amp")
        self.assertEqual((a.epochs, a.warmup_epochs, a.use_amp), (3, 1, False))

    def test_brant(self):
        from experiments.run_brant_regression import build_argparser
        a = build_argparser().parse_args([])
        self.assertEqual((a.train_mode, a.context_patches, a.stride_s, a.patch_pool),
                         ("per_subject", 1, 0.5, "mean"))
        self.assertEqual((a.head, a.epochs, a.early_stop_patience, a.extract_batch),
                         ("linear", 200, 30, 16))
        self.assertEqual((a.lr, a.weight_decay, a.warmup_epochs, a.min_lr, a.batch_size),
                         (1e-3, 1e-4, 10, 1e-6, 256))
        self.assertTrue(a.use_amp)

    def test_paper_cells_land_in_paper_folders(self):
        """A paper folder takes only a run with every recorded setting, all nine
        subjects and no cap; any departure is named, a smoke run goes elsewhere."""
        from experiments import run_brant_regression as rb
        from experiments import run_ieeg_fm_regression as rr
        from experiments.common import STANFORD_SUBJECTS

        def where(runner, args, cuda=True):
            # --device auto resolves to a GPU here unless cuda=False
            with mock.patch("torch.cuda.is_available", return_value=cuda):
                runner.resolve_device(args)
            with mock.patch.dict(os.environ, {"CORTEG_OUTPUT_ROOT": "/o"}):
                parts = Path(runner.resolve_save_root(args)).relative_to("/o").parts
            return parts[0], parts[1]              # (top folder, adaptation tag)

        paper, smoke = "ieeg_fm_regression", "ieeg_fm_regression_smoke"
        for argv, tag in (
                ([], "probe"),
                (["--mode", "finetune"], "ft"),
                (["--head", "temporal"], "temporal"),
                (["--subjects", ",".join(STANFORD_SUBJECTS)], "probe"),
                (["--device", "cuda", "--emb_cache", "/c", "--data_root", "/d"], "probe"),
                (["--device", "cpu"], "probe_use_amp=False"),          # fp32, not the paper's
                (["--train_mode", "pooled", "--reseed_pooled_head"], "probe_reseed"),
                (["--bb_pool", "last10"], "probe_last10"),
                (["--head", "temporal", "--temporal_direction", "uni"], "temporal_uni"),
                (["--reref", "car"], "probe_reref=car"),
                (["--epochs", "2"], "probe_epochs=2"),
                (["--lr", "3e-4"], "probe_lr=0.0003"),
                (["--no_amp"], "probe_use_amp=False"),
                (["--mode", "finetune", "--unfreeze_last_n", "6"], "ft_unfreeze_last_n=6"),
                (["--mode", "finetune", "--ft_lr", "1e-3"], "ft_ft_lr=0.001"),
                (["--mode", "finetune", "--warmup_epochs", "10"], "ft_warmup_epochs=10"),
                (["--head", "temporal", "--seq_len", "16"], "temporal_seq_len=16"),
                (["--head", "temporal", "--epochs", "60"], "temporal_epochs=60")):
            self.assertEqual(where(rr, self._reg(*argv)), (paper, tag), argv)
        self.assertEqual(where(rr, self._reg(), cuda=False), (paper, "probe_use_amp=False"))
        for argv, tag in ((["--max_windows", "48"], "probe_w48"),
                          (["--subjects", "mv"], "probe_subs=mv"),
                          (["--subjects", "wm,mv", "--max_windows", "32"], "probe_subs=mv+wm_w32"),
                          (["--skip_subjects", "mv"], "probe_subs=bp+cc+ht+jc+jp+wc+wm+zt"),
                          (["--max_windows", "48", "--subjects", "mv", "--epochs", "2"],
                           "probe_epochs=2_subs=mv_w48")):
            self.assertEqual(where(rr, self._reg(*argv)), (smoke, tag), argv)

        p = rb.build_argparser()
        for argv, tag in (([], "brant"), (["--extract_batch", "8"], "brant"),
                          (["--stride_s", "0.1"], "brant_s0.1"),
                          (["--context_patches", "15", "--patch_pool", "last"], "brant_L15_last"),
                          (["--epochs", "5"], "brant_epochs=5"),
                          (["--head", "mlp", "--mlp_hidden", "64"], "brant_mlp_mlp_hidden=64")):
            self.assertEqual(where(rb, p.parse_args(argv)), (paper, tag), argv)
        self.assertEqual(where(rb, p.parse_args(["--device", "cpu"])),
                         (paper, "brant_use_amp=False"))
        for argv, tag in ((["--max_windows", "60"], "brant_w60"),
                          (["--max_anchors", "3"], "brant_a3"),
                          (["--subjects", "mv,wm"], "brant_subs=mv+wm")):
            self.assertEqual(where(rb, p.parse_args(argv)), (smoke, tag), argv)

    def test_a_cpu_run_is_named_apart_from_a_gpu_run(self):
        """AMP is applied on CUDA only, so a CPU run records it off, together with
        the device it used; its folder, and what --skip_if_done compares, differ."""
        from experiments import run_brant_regression as rb
        from experiments import run_ieeg_fm_regression as rr
        for runner, parse, rest in (
                (rr, lambda a: self._reg(*a), rr.NOT_SETTINGS),
                (rb, lambda a: rb.build_argparser().parse_args(a), rb.BRANT_NOT_SETTINGS)):
            got = {}
            for dev in ("cpu", "cuda"):
                args = parse(["--device", dev])
                runner.resolve_device(args)
                self.assertEqual((args.device, args.use_amp), (dev, dev == "cuda"))
                got[dev] = (runner.adaptation_tag(args), rr.run_settings(args, rest))
            self.assertNotEqual(got["cpu"][0], got["cuda"][0])
            self.assertIn("use_amp=False", got["cpu"][0])
            self.assertEqual(got["cpu"][1]["device"], "cpu")
            self.assertFalse(rr.same_settings(got["cpu"][1], got["cuda"][1]))
            # --no_amp on a GPU is named, but is still not the CPU run
            args = parse(["--device", "cuda", "--no_amp"])
            runner.resolve_device(args)
            self.assertEqual(runner.adaptation_tag(args), got["cpu"][0])
            self.assertFalse(rr.same_settings(rr.run_settings(args, rest), got["cpu"][1]))

    def test_joint_finetune_is_refused(self):
        from experiments.run_ieeg_fm_regression import main
        with self.assertRaises(SystemExit):
            main(["--fm", "brainbert", "--mode", "finetune", "--head", "temporal"])

    def test_reseed_is_for_the_pooled_probe_only(self):
        from experiments.run_ieeg_fm_regression import main
        for extra in (["--train_mode", "per_subject"], ["--train_mode", "loo"],
                      ["--train_mode", "pooled", "--head", "temporal"],
                      ["--train_mode", "pooled", "--mode", "finetune"]):
            with self.assertRaises(SystemExit) as cm:
                main(["--fm", "popt", "--reseed_pooled_head", *extra])
            self.assertIn("pooled frozen probe", str(cm.exception), extra)

    def test_loo_needs_two_subjects(self):
        from experiments.run_ieeg_fm_regression import main
        for mode in ("probe", "finetune"):
            with self.assertRaises(SystemExit) as cm:
                main(["--fm", "popt", "--mode", mode, "--train_mode", "loo", "--subjects", "mv"])
            self.assertIn("two subjects", str(cm.exception))


class TestRunBookkeeping(unittest.TestCase):
    """Skip-if-done and LOO stage-1 reuse must only ever pick up the same run."""

    def test_skip_if_done_needs_the_exact_run(self):
        from experiments import run_ieeg_fm_regression as rr
        from experiments.common import STANFORD_SUBJECTS
        argv = ["--fm", "popt", "--train_mode", "per_subject"]
        with tempfile.TemporaryDirectory() as d:
            common = ["--save_root", d, "--skip_if_done", "--native_root", os.path.join(d, "none")]
            # a CPU run's result, recorded as main() records it
            args = rr.resolve_budget(rr.build_argparser().parse_args(argv + common
                                                                     + ["--device", "cpu"]))
            rr.resolve_device(args)
            per = {s: {"corr_mean": 0.05, "mse": 1.0} for s in STANFORD_SUBJECTS}
            rr.write_results(os.path.join(d, "results_persub.json"), "per_subject",
                             0.05, 1.0, per, args)
            # The same run returns the recorded score without reading any data...
            self.assertAlmostEqual(rr.main(argv + common + ["--device", "cpu"]), 0.05)
            # ...anything else runs (and fails here on the missing native files),
            # a GPU run of the paper's settings included.
            for extra in (["--epochs", "3"], ["--reref", "car"], ["--seed", "0"],
                          ["--subjects", "bp,cc"], ["--max_windows", "48"]):
                with self.assertRaises(FileNotFoundError, msg=str(extra)):
                    rr.main(argv + common + ["--device", "cpu"] + extra)
            with self.assertRaises(FileNotFoundError):
                rr.main(argv + common + ["--device", "cuda"])
            # AMP is off on a CPU either way, so --no_amp there is the same run
            self.assertAlmostEqual(rr.main(argv + common + ["--device", "cpu", "--no_amp"]), 0.05)
            # A file that records no settings is not trusted.
            with open(os.path.join(d, "results_persub.json"), "w") as f:
                json.dump({"per_subject": per, "score": 0.05}, f)
            with self.assertRaises(FileNotFoundError):
                rr.main(argv + common + ["--device", "cpu"])

    def test_brant_skip_if_done(self):
        from experiments import run_brant_regression as rb
        from experiments import run_ieeg_fm_regression as rr
        from experiments.common import STANFORD_SUBJECTS
        def settings(argv):
            args = rb.build_argparser().parse_args(argv)
            rb.resolve_device(args)
            return args, rr.run_settings(args, rb.BRANT_NOT_SETTINGS)

        with tempfile.TemporaryDirectory() as d:
            args, want = settings(["--save_root", d, "--device", "cpu"])
            per = {s: {"corr_mean": 0.03, "mse": 1.0} for s in STANFORD_SUBJECTS}
            rb.write_results(os.path.join(d, "results_persub.json"), args, "per_subject", per, {})
            self.assertTrue(rr.finished_result(d, "per_subject", want, STANFORD_SUBJECTS))
            for argv in (["--max_anchors", "3"], ["--stride_s", "0.1"], ["--epochs", "5"]):
                other = settings(argv + ["--device", "cpu"])[1]
                self.assertIsNone(rr.finished_result(d, "per_subject", other, STANFORD_SUBJECTS))
            # a GPU run is not the CPU run on file
            self.assertIsNone(rr.finished_result(d, "per_subject", settings(["--device", "cuda"])[1],
                                                 STANFORD_SUBJECTS))
            # the extraction batch is a memory knob, not a setting
            eb = settings(["--extract_batch", "4", "--device", "cpu"])[1]
            self.assertTrue(rr.finished_result(d, "per_subject", eb, STANFORD_SUBJECTS))

    def test_reseeded_pooled_probe_ignores_the_rng_state(self):
        """Extraction on a cold cache draws from the RNG before the pooled head is
        built. By default (the paper runs) that changes the head; reseeded, not."""
        import torch
        from experiments import run_ieeg_fm_regression as rr
        from experiments.common import set_seed
        rng = np.random.RandomState(0)
        r = lambda n, k: rng.randn(n, k).astype(np.float32)  # noqa: E731
        subs = [{"sub": s, "emb_tr": r(60, 6), "y_tr": r(60, 5), "emb_va": r(10, 6),
                 "y_va": r(10, 5), "emb_te": r(20, 6), "y_te": r(20, 5)} for s in ("bp", "cc")]

        def run(d, reseed, draws):
            args = rr.resolve_budget(rr.build_argparser().parse_args(
                ["--fm", "popt", "--train_mode", "pooled", "--epochs", "2"]
                + (["--reseed_pooled_head"] if reseed else [])))
            set_seed(42)
            torch.randn(draws)                    # what building the FMs would draw
            return rr.run_pooled(subs, 5, args, torch.device("cpu"), d)[1]

        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(run(d, True, 0), run(d, True, 1000))
            self.assertNotEqual(run(d, False, 0), run(d, False, 1000))

    def test_loo_stage1_is_reused_only_by_the_same_run(self):
        import torch
        from experiments import run_ieeg_fm_regression as rr
        rng = np.random.RandomState(0)

        def subject(name):
            r = lambda n, k: rng.randn(n, k).astype(np.float32)  # noqa: E731
            return {"sub": name, "emb_tr": r(40, 6), "y_tr": r(40, 5), "emb_va": r(10, 6),
                    "y_va": r(10, 5), "emb_te": r(20, 6), "y_te": r(20, 5)}

        subs = [subject("bp"), subject("cc"), subject("ht")]
        stage1 = []
        real = rr.train_head

        def counting(*a, **k):
            stage1.append(k.get("init_state") is None)
            return real(*a, **k)

        def run(d, argv, subjects):
            stage1.clear()
            args = rr.resolve_budget(rr.build_argparser().parse_args(
                ["--fm", "popt", "--train_mode", "loo", "--epochs", "1", *argv]))
            rr.run_loo(subjects, 5, args, torch.device("cpu"), d)
            return sum(stage1)

        rr.train_head = counting
        try:
            with tempfile.TemporaryDirectory() as d:
                self.assertEqual(run(d, [], subs), 3)                    # fresh
                self.assertEqual(run(d, [], subs), 0)                    # same run: reused
                self.assertEqual(run(d, ["--epochs", "2"], subs), 3)     # other settings
                self.assertEqual(run(d, ["--epochs", "2"], subs[:2]), 2)  # other subjects
                with open(os.path.join(d, "_loo_done.json")) as f:
                    self.assertEqual(json.load(f)["args"]["epochs"], 2)  # the summary says which run
        finally:
            rr.train_head = real


class TestScript(unittest.TestCase):
    """Every call in the script must parse under the runner it calls."""

    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def _parse_all(self, fn, parser, extra):
        calls = _script_calls(self.text, fn)
        self.assertTrue(calls, f"no {fn} calls in {SCRIPT.name}")
        parsed = []
        for toks in calls:
            flags = toks[1:]                      # toks[0] is the save folder
            self.assertFalse(toks[0].startswith("--"), f"{fn} call lacks a save folder")
            try:
                parsed.append(parser.parse_args(flags + extra))
            except SystemExit:
                self.fail(f"{fn} call does not parse: {' '.join(flags)}")
        return parsed

    WRAPPER_FLAGS = ["--skip_if_done", "--native_root", "x", "--emb_cache", "x", "--save_root", "x"]

    def _same_folder_as_runner(self, calls, parsed, tag_fn, fm_of):
        """The script's save folders are the runners' own default layout."""
        for toks, a in zip(calls, parsed):
            want = "/".join(["/x", tag_fn(a), "Stanford", fm_of(a), a.train_mode, f"seed{a.seed}"])
            self.assertEqual(toks[0], want)

    def test_fm_calls_parse(self):
        from experiments.run_ieeg_fm_regression import (adaptation_tag, build_argparser,
                                                        resolve_budget)
        parsed = [resolve_budget(a) for a in self._parse_all("reg", build_argparser(),
                                                             self.WRAPPER_FLAGS)]
        cells = {(a.fm, a.mode, a.head, a.train_mode, a.seed) for a in parsed}
        for fm in ("brainbert", "popt"):
            # the Table 1 cells
            self.assertIn((fm, "probe", "temporal", "per_subject", 42), cells)
            self.assertIn((fm, "probe", "linear", "loo", 42), cells)
            self.assertIn((fm, "finetune", "linear", "loo", 42), cells)
        for a in parsed:                          # the script states the budgets it uses
            want = {"probe": (60, 10), "finetune": (60, 5)}.get(a.mode)
            if a.head == "temporal":
                want = (100, 10)
            self.assertEqual((a.epochs, a.warmup_epochs), want)
            self.assertEqual(a.early_stop_patience, 20)
        self._same_folder_as_runner(_script_calls(self.text, "reg"), parsed,
                                    adaptation_tag, lambda a: a.fm)

    def test_brant_calls_parse(self):
        from experiments.run_brant_regression import adaptation_tag, build_argparser
        parsed = self._parse_all("brant", build_argparser(), self.WRAPPER_FLAGS)
        strides = {a.stride_s for a in parsed if a.train_mode == "per_subject"}
        self.assertEqual(strides, {0.5, 0.1})
        for a in parsed:
            self.assertEqual((a.context_patches, a.epochs, a.early_stop_patience),
                             (1, 200, 30))
        self._same_folder_as_runner(_script_calls(self.text, "brant"), parsed,
                                    adaptation_tag, lambda a: "brant")

    def test_every_runner_call_goes_through_a_wrapper(self):
        """The wrappers are the only place the runners are called."""
        body = [l for l in self.text.splitlines() if not l.lstrip().startswith("#")]
        n = sum("python -m experiments." in l for l in body)
        self.assertEqual(n, 2)


class TestScriptRuns(unittest.TestCase):
    """Runs the script with a stand-in `python` that only records its arguments."""

    SUBJECTS = ("bp", "cc", "ht", "jc", "jp", "mv", "wc", "wm", "zt")

    def _run(self, args, env_extra=None, native=True, cwd=None):
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        bindir, log = os.path.join(d, "bin"), os.path.join(d, "calls.log")
        os.makedirs(bindir)
        fake = os.path.join(bindir, "python")
        with open(fake, "w") as f:           # `python -c` is the script's CUDA check
            f.write('#!/bin/sh\nif [ "$1" = "-c" ]; then exit "${FAKE_CUDA_RC:-0}"; fi\n'
                    'pwd >> "$FAKE_LOG.pwd"\n'
                    'for a in "$@"; do printf \'%s\\037\' "$a"; done >> "$FAKE_LOG"\n'
                    'printf \'\\n\' >> "$FAKE_LOG"\n')
        os.chmod(fake, os.stat(fake).st_mode | stat.S_IEXEC)
        nat = os.path.join(d, "data", "native_1k", "built")
        os.makedirs(nat)
        if native:
            for s in self.SUBJECTS:
                open(os.path.join(nat, f"{s}_native1k.npz"), "w").close()
        env = {k: v for k, v in os.environ.items()
               if k not in ("CORTEG_NATIVE1K_ROOT", "ONLY_TABLE1", "RUN_BB_FT_POOLED",
                            "ECOG_OUTPUT_ROOT", "ECOG_DATA_ROOT")}
        env.update({"PATH": bindir + os.pathsep + env["PATH"], "FAKE_LOG": log,
                    "CORTEG_OUTPUT_ROOT": os.path.join(d, "out"),
                    "CORTEG_DATA_ROOT": os.path.join(d, "data")})
        env.update(env_extra or {})
        res = subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True,
                             text=True, cwd=cwd or str(REPO))
        calls = []
        if os.path.exists(log):
            with open(log) as f:
                calls = [line.rstrip("\n").split("\x1f")[:-1] for line in f if line.strip()]
            with open(log + ".pwd") as f:        # the runners start from the repository root
                self.assertEqual({line.strip() for line in f}, {str(REPO)})
        return res, calls, d

    @staticmethod
    def _flag(call, name):
        return call[call.index(name) + 1]

    def test_paper_run(self):
        res, calls, d = self._run([])
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(len(calls), 31)          # 34 with RUN_BB_FT_POOLED=1
        out = os.path.join(d, "out", "ieeg_fm_regression")
        for c in calls:
            self.assertIn("--skip_if_done", c)
            self.assertTrue(self._flag(c, "--save_root").startswith(out + os.sep))
            self.assertEqual(self._flag(c, "--emb_cache"), os.path.join(out, "fm_emb_cache"))
            self.assertEqual(self._flag(c, "--native_root"),
                             os.path.join(d, "data", "native_1k", "built"))
        bb_ft_pooled = [c for c in calls if "brainbert" in c and "finetune" in c
                        and self._flag(c, "--train_mode") == "pooled"]
        self.assertEqual(bb_ft_pooled, [])        # not a paper cell; opt-in only
        res, calls, _ = self._run([], {"RUN_BB_FT_POOLED": "1"})
        self.assertEqual(len(calls), 34)
        res, calls, _ = self._run([], {"ONLY_TABLE1": "1"})
        self.assertEqual(len(calls), 3)

    def _assert_smoke(self, calls, d, tail=()):
        smoke = os.path.join(d, "out", "ieeg_fm_regression_smoke")
        self.assertTrue(calls)
        for c in calls:
            self.assertTrue(self._flag(c, "--save_root").startswith(smoke + os.sep))
            self.assertEqual(self._flag(c, "--emb_cache"), os.path.join(smoke, "fm_emb_cache"))
            if tail:
                self.assertEqual(c[-len(tail):], list(tail))
        self.assertFalse(os.path.exists(os.path.join(d, "out", "ieeg_fm_regression")))

    def test_smoke_pass_writes_elsewhere(self):
        args = ["--subjects", "mv,wm", "--max_windows", "5", "--epochs", "1", "--device", "cpu"]
        res, calls, d = self._run(args, native=False)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(len(calls), 31)
        self._assert_smoke(calls, d, args)

    def test_one_subject_skips_the_loo_cells(self):
        """Leave-one-subject-out needs two subjects; the rest of the pass still runs."""
        for args in (["--subjects", "mv", "--max_windows", "5"],
                     ["--subjects", "mv,wm", "--skip_subjects", "wm"]):
            res, calls, d = self._run(args, native=False)
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertEqual(len(calls), 27, args)                # 31 minus the four LOO cells
            self.assertFalse([c for c in calls if self._flag(c, "--train_mode") == "loo"])
            self.assertEqual(res.stdout.count("leave-one-subject-out needs"), 4)
            self._assert_smoke(calls, d, args)

    def test_a_cpu_pass_is_a_smoke_pass(self):
        """The paper ran on GPUs with AMP; a CPU pass must not fill its folders or caches."""
        res, calls, d = self._run(["--device", "cpu"])
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(len(calls), 31)
        self._assert_smoke(calls, d, ["--device", "cpu"])
        res, calls, d = self._run([], {"FAKE_CUDA_RC": "1"})    # auto, and torch sees no GPU
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("no CUDA device", res.stdout)
        self._assert_smoke(calls, d)
        res, calls, d = self._run(["--device", "cuda"], {"FAKE_CUDA_RC": "1"})
        self.assertEqual(res.returncode, 0, res.stderr)     # an explicit GPU run is not probed
        out = os.path.join(d, "out", "ieeg_fm_regression")
        self.assertTrue(all(self._flag(c, "--save_root").startswith(out + os.sep) for c in calls))

    def test_runs_from_any_directory(self):
        elsewhere = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(elsewhere, ignore_errors=True))
        nat = os.path.join(elsewhere, "mydata", "native_1k", "built")
        os.makedirs(nat)
        for s in self.SUBJECTS:
            open(os.path.join(nat, f"{s}_native1k.npz"), "w").close()
        res, calls, _ = self._run(["--data_root", "mydata"], {"ONLY_TABLE1": "1"},
                                  native=False, cwd=elsewhere)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(len(calls), 3)
        for c in calls:
            self.assertEqual(self._flag(c, "--native_root"), nat)
            self.assertEqual(self._flag(c, "--data_root"), os.path.join(elsewhere, "mydata"))

    def test_data_root_locates_the_native_files(self):
        d2 = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d2, ignore_errors=True))
        nat = os.path.join(d2, "native_1k", "built")
        os.makedirs(nat)
        for s in self.SUBJECTS:
            open(os.path.join(nat, f"{s}_native1k.npz"), "w").close()
        res, calls, _ = self._run(["--data_root", d2], {"ONLY_TABLE1": "1"}, native=False)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertTrue(all(self._flag(c, "--native_root") == nat for c in calls))

    def test_other_flags_are_refused(self):
        for bad in (["--reref", "car"], ["--bb_pool", "last10"], ["--max_anchors", "3"],
                    ["--subjects"], ["--device", "gpu"]):
            res, calls, _ = self._run(bad)
            self.assertEqual(res.returncode, 2, bad)
            self.assertEqual(calls, [], bad)


class TestNativeWindows(unittest.TestCase):

    def test_windows_end_at_the_target(self):
        from data.stanford_native import build_native_windows
        T, C = 5000, 3
        stream = np.arange(T * C, dtype=np.float32).reshape(T, C)
        edges = np.array([999, 1039, 4999])
        X = build_native_windows(stream, edges, 1000)
        self.assertEqual(X.shape, (3, C, 1000))
        for n, s in enumerate(edges):
            np.testing.assert_array_equal(X[n, :, -1], stream[s])          # edge included
            np.testing.assert_array_equal(X[n, :, 0], stream[s - 999])

    def test_out_of_range_windows_raise(self):
        from data.stanford_native import build_native_windows
        stream = np.zeros((2000, 2), np.float32)
        for bad in (998, 2000):
            with self.assertRaises(ValueError):
                build_native_windows(stream, np.array([bad]), 1000)

    def test_native_file_pairs_with_the_pickle(self):
        """Window n must pair with target n, and a count mismatch must stop."""
        from data.stanford_native import NativeSubject
        rng = np.random.RandomState(0)
        C, n_tr, n_te = 4, 6, 3
        with tempfile.TemporaryDirectory() as d:
            y_tr = rng.randn(n_tr, 5).astype(np.float32)
            y_te = rng.randn(n_te, 5).astype(np.float32)
            obj = (np.zeros((n_tr, C, 200, 2)), y_tr, np.zeros((n_te, C, 200, 2)), y_te,
                   np.zeros((n_tr, C, 128)), np.zeros((n_te, C, 128)))
            with open(os.path.join(d, "zz_features.pkl"), "wb") as f:
                pickle.dump(obj, f)
            stream = rng.randn(4000, C).astype(np.float32)
            np.savez(os.path.join(d, "zz_native1k.npz"), data_kept=stream,
                     win_starts_tr=1039 + 40 * np.arange(n_tr),
                     win_starts_te=3000 + 40 * np.arange(n_te),
                     xyz=rng.randn(C, 3).astype(np.float32), fs=np.int64(1000),
                     win_len=np.int64(1000))
            ns = NativeSubject("zz", d, d)
            got_tr, got_te = ns.targets()
            np.testing.assert_array_equal(got_tr, y_tr)
            X = ns.windows("test", nmax=2)
            self.assertEqual(X.shape, (2, C, 1000))
            np.testing.assert_array_equal(X[1, :, -1], stream[3040])

            np.savez(os.path.join(d, "yy_native1k.npz"), data_kept=stream,
                     win_starts_tr=1039 + 40 * np.arange(n_tr + 1),
                     win_starts_te=3000 + 40 * np.arange(n_te),
                     xyz=rng.randn(C, 3).astype(np.float32))
            with open(os.path.join(d, "yy_features.pkl"), "wb") as f:
                pickle.dump(obj, f)
            with self.assertRaises(AssertionError):
                NativeSubject("yy", d, d).targets()

    def test_missing_file_says_how_to_build_it(self):
        from data.stanford_native import NativeSubject
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError) as cm:
                NativeSubject("bp", d, d)
        self.assertIn("python -m data.stanford_native", str(cm.exception))

    def test_split_rule(self):
        from data.stanford_native import split_boundary
        self.assertEqual(split_boundary(610_040), (400_000, "first400s"))
        self.assertEqual(split_boundary(178_960), (119_306, "first2/3"))

    def test_roots_follow_the_data_root(self):
        """--native_root > $CORTEG_NATIVE1K_ROOT > <data_root>/native_1k/built, with the
        runner's --data_root as data_root; the raw download sits where the tutorial says."""
        from data.stanford_native import native_root, raw_root
        env = {k: v for k, v in os.environ.items()
               if k not in ("CORTEG_NATIVE1K_ROOT", "CORTEG_STANFORD_RAW_ROOT")}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(native_root("", "/d"), os.path.join("/d", "native_1k", "built"))
            self.assertEqual(native_root("/n", "/d"), "/n")
            self.assertEqual(raw_root("", "/d"), os.path.join("/d", "raw", "Stanford"))
            os.environ["CORTEG_NATIVE1K_ROOT"] = "/e"
            self.assertEqual(native_root("", "/d"), "/e")

    def test_builder_writes_where_the_runners_look(self):
        """A build with --data_root D is found by a run with --data_root D."""
        from data import stanford_native as sn
        env = {k: v for k, v in os.environ.items()
               if k not in ("CORTEG_NATIVE1K_ROOT", "CORTEG_STANFORD_RAW_ROOT")}
        seen = []

        def fake(sub, raw_dir, pkl_root, out_root, verbose=True):
            seen.append((raw_dir, pkl_root, out_root))
            return {"subject": sub, "wrote": True}

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(sn, "process_subject", fake):
            for flag in ("--data_root", "--pkl_root"):
                seen.clear()
                self.assertEqual(sn.main(["--subjects", "mv", flag, d]), 0)
                self.assertEqual(seen, [(os.path.join(d, "raw", "Stanford"), d,
                                         sn.native_root("", d))])
                self.assertEqual(seen[0][2], os.path.join(d, "native_1k", "built"))

    def test_builder_takes_comma_or_space_separated_subjects(self):
        from data.stanford_native import parse_subjects
        self.assertEqual(parse_subjects(["bp,mv", "wc"]), ["bp", "mv", "wc"])
        self.assertEqual(parse_subjects(["mv", "mv"]), ["mv"])
        for bad in (["mv,zz"], ["zz"]):
            with self.assertRaises(SystemExit):
                parse_subjects(bad)


class TestBrantAnchors(unittest.TestCase):

    def test_no_window_sees_past_its_target(self):
        from experiments.run_brant_regression import anchor_plan
        fs, stride = 1000.0, 0.5
        edges = 1039 + 40 * np.arange(3000)
        end, bin_id, anchor_end = anchor_plan(edges, fs, stride, n250=10 ** 9)
        for n in range(len(edges)):
            a = anchor_end[int(bin_id[n])]
            self.assertLessEqual(a, end[n])                        # causal
            self.assertLess(end[n] - a, int(stride * 250))         # at most one bin stale
        self.assertLess(len(anchor_end), len(edges) / 10)          # coarse grid

    def test_windows_in_a_bin_share_one_embedding(self):
        from experiments.run_brant_regression import extract_split
        calls = []

        def fake_emb(x, power, et, ec):          # (B, C, L, 1500) -> (B, C, L, 2048)
            calls.append(x.shape[0])
            v = x[..., -1:].mean(dim=(1, 2, 3), keepdim=True)     # last sample of the context
            return v.expand(x.shape[0], x.shape[1], x.shape[2], 2048).clone()

        fs = 1000.0
        stream = np.arange(40_000, dtype=np.float32)[:, None].repeat(2, axis=1)
        edges = 1039 + 40 * np.arange(800)
        args = SimpleNamespace(context_patches=1, stride_s=0.5, max_anchors=0,
                               extract_batch=16, patch_pool="mean")
        emb, meta = extract_split(None, None, stream, fs, edges, args, "cpu", emb_fn=fake_emb)
        self.assertEqual(emb.shape, (800, 2048))
        self.assertEqual(meta["n_anchors"], len(np.unique(emb[:, 0])))
        self.assertTrue(all(b <= 16 for b in calls))
        # the embedding carries the context's last sample, which never lies
        # after the window's own target
        target_250 = np.round(edges * 0.25)
        self.assertTrue(np.all(emb[:, 0] / 4.0 <= target_250 + 1))

    def test_short_history_is_tiled_not_zeroed(self):
        from ieeg_fm import BRANT_PATCH_LEN, _context_patches
        stream = np.arange(1, 501, dtype=np.float32)[:, None]     # 2 s at 250 Hz
        p, padded = _context_patches(stream, 500, 1)
        self.assertTrue(padded)
        self.assertEqual(p.shape, (1, 1, BRANT_PATCH_LEN))
        self.assertGreater(p.min(), 0)                            # no zeros
        self.assertEqual(p[0, 0, -1], 500)                        # still ends at the anchor

    def test_cache_names_match_the_paper_runs(self):
        from experiments.run_brant_regression import _cache_path
        a = SimpleNamespace(max_windows=0, max_anchors=0, patch_pool="mean",
                            context_patches=1, stride_s=0.5)
        self.assertEqual(os.path.basename(_cache_path("c", "mv", "train", a)),
                         "brant_fair_Stanford_mv_train_L1_s0.5.npz")
        a.stride_s = 0.1
        self.assertEqual(os.path.basename(_cache_path("c", "mv", "test", a)),
                         "brant_fair_Stanford_mv_test_L1_s0.1.npz")
        a.context_patches, a.patch_pool = 15, "last"
        self.assertEqual(os.path.basename(_cache_path("c", "mv", "test", a)),
                         "brant_fair_Stanford_mv_test_L15_s0.1_plast.npz")

    def test_debug_caps_change_the_cache_name(self):
        from experiments.run_brant_regression import _cache_path
        a = SimpleNamespace(max_windows=7, max_anchors=0, patch_pool="mean",
                            context_patches=1, stride_s=0.5)
        self.assertEqual(os.path.basename(_cache_path("c", "mv", "train", a)),
                         "brant_fair_Stanford_mv_train_L1_s0.5_w7.npz")
        a.max_anchors = 3
        self.assertEqual(os.path.basename(_cache_path("c", "mv", "train", a)),
                         "brant_fair_Stanford_mv_train_L1_s0.5_w7_a3.npz")

    def test_a_cache_from_another_run_is_refused(self):
        """The paper runs' own script named only the anchor cap, so a capped array
        can sit under the uncapped name; it must not be read as the full split."""
        from experiments import run_brant_regression as rb
        with tempfile.TemporaryDirectory() as d:
            ns = SimpleNamespace(sub="mv", fs=1000.0, right_edges=lambda split: np.arange(100))

            def write(args, rows):
                for split in ("train", "test"):
                    np.savez(rb._cache_path(d, "mv", split, args),
                             emb=np.zeros((rows, 2048), np.float32),
                             y=np.zeros((rows, 5), np.float32), meta=json.dumps({"n": rows}))

            full = rb.build_argparser().parse_args(["--emb_cache", d])
            write(full, 7)
            with self.assertRaises(RuntimeError):
                rb.extract_subject(lambda: (None, None), ns, full, "cpu")
            capped = rb.build_argparser().parse_args(["--emb_cache", d, "--max_windows", "7"])
            write(capped, 7)                      # a capped run reads its own, capped file
            emb_tr, y_tr, emb_te, y_te, _ = rb.extract_subject(lambda: (None, None), ns,
                                                               capped, "cpu")
            self.assertEqual((emb_tr.shape[0], y_te.shape[0]), (7, 7))


class TestBrainBERTAndPopT(unittest.TestCase):

    def test_cache_names_match_the_paper_runs(self):
        from experiments.run_ieeg_fm_regression import _cache_path, _frontend_cache_path
        n = lambda p: os.path.basename(p)  # noqa: E731
        self.assertEqual(n(_cache_path("c", "brainbert", "mv", "train", 1000.0, "laplacian_xyz")),
                         "brainbert_Stanford_mv_train_fs1000_laplacian_xyz.npz")
        self.assertEqual(n(_frontend_cache_path("c", "popt", "bp", "test", 1000.0,
                                                "laplacian_xyz")),
                         "popt_frontend_Stanford_bp_test_fs1000_laplacian_xyz.npz")
        self.assertEqual(n(_cache_path("c", "popt", "mv", "train", 1000.0, "laplacian_xyz",
                                       nmax=300, bb_pool="last10")),
                         "popt_Stanford_mv_train_fs1000_laplacian_xyz_last10_n300.npz")
        # BrainBERT's front end is the spectrogram, which pooling does not touch
        self.assertEqual(n(_frontend_cache_path("c", "brainbert", "mv", "train", 1000.0,
                                                "laplacian_xyz", bb_pool="last10")),
                         "brainbert_frontend_Stanford_mv_train_fs1000_laplacian_xyz.npz")

    def test_a_cache_from_another_run_is_refused(self):
        from experiments import run_ieeg_fm_regression as rr
        rng = np.random.RandomState(0)
        ns = SimpleNamespace(sub="mv", fs=1000.0, right_edges=lambda split: np.arange(100),
                             xyz_mm=(rng.randn(4, 3) * 20).astype(np.float32))
        with tempfile.TemporaryDirectory() as d:
            args = rr.resolve_budget(rr.build_argparser().parse_args(["--fm", "popt"]))
            np.savez(rr._cache_path(d, "popt", "mv", "train", 1000.0, "laplacian_xyz"),
                     emb=np.zeros((7, 512), np.float32), y=np.zeros((7, 5), np.float32))
            with self.assertRaises(RuntimeError):
                rr.get_subject_embeddings("popt", ns, args, "cpu", d)
            args.mode = "finetune"
            np.savez(rr._frontend_cache_path(d, "popt", "mv", "train", 1000.0, "laplacian_xyz"),
                     fe=np.zeros((7, 4, 768), np.float32))
            with self.assertRaises(RuntimeError):
                rr.get_subject_frontend("popt", ns, args, "cpu", d)

    def test_centre_pooling_sits_half_a_second_before_the_target(self):
        """The timing the runner's docstring states, computed from BrainBERT's STFT."""
        from scipy import signal as sps
        import ieeg_fm
        from experiments.run_ieeg_fm_regression import bb_pool_slice
        n = ieeg_fm.PRETRAIN_FS                                   # a 1 s window, resampled
        _, t, _ = sps.stft(np.zeros(n), ieeg_fm.PRETRAIN_FS, nperseg=ieeg_fm.STFT_NPERSEG,
                           noverlap=ieeg_fm.STFT_NOVERLAP)
        clip = ieeg_fm.STFT_ZSCORE_CLIP
        t = t[clip:-clip]                                         # BrainBERT's edge clip
        self.assertEqual(len(t), ieeg_fm._stft_spec(np.random.randn(n)).shape[0])
        lag = lambda mode: 1.0 - t[bb_pool_slice(len(t), mode)].mean()  # noqa: E731
        self.assertAlmostEqual(lag("center10"), 0.50, delta=0.01)
        self.assertAlmostEqual(lag("last10"), 0.35, delta=0.01)
        centre = t[bb_pool_slice(len(t), "center10")]
        self.assertAlmostEqual(centre[0], 0.39, delta=0.01)
        self.assertAlmostEqual(centre[-1], 0.61, delta=0.01)

    def test_chunked_extraction_equals_one_call(self):
        """Extraction is chunked to bound memory; it must not change a value."""
        import torch
        import ieeg_fm
        from experiments import run_ieeg_fm_regression as rr

        class FakeBrainBERT(torch.nn.Module):     # per-spectrogram, like the real one
            def __init__(self):
                super().__init__()
                g = torch.Generator().manual_seed(0)
                self.w = torch.randn(ieeg_fm.INPUT_DIM, ieeg_fm.HIDDEN_DIM, generator=g)

            def forward(self, x, mask, intermediate_rep=True):
                return torch.tanh(x @ self.w) + torch.arange(x.shape[1])[None, :, None]

        fake = FakeBrainBERT()
        rng = np.random.RandomState(0)
        x = rng.randn(70, 4, 1000).astype(np.float32)
        xyz = rng.randn(4, 3).astype(np.float32) * 20
        saved = (ieeg_fm.load_brainbert, rr.EXTRACT_CHUNK)
        ieeg_fm.load_brainbert = lambda *a, **k: fake
        rr.EXTRACT_CHUNK = 32                     # three chunks, the last one ragged
        try:
            for mode, pool in (("center10", "default"), ("last10", "none")):
                got = rr.brainbert_per_electrode(x, 1000.0, xyz, "laplacian_xyz", "cpu", mode)
                ref = ieeg_fm.brainbert_embeddings(x, fs=1000.0, xyz_mm=xyz, device="cpu",
                                                   reref="laplacian_xyz", pool=pool, model=fake)
                if pool == "none":
                    ref = ref[:, :, rr.bb_pool_slice(ref.shape[2], mode)].mean(axis=2)
                self.assertEqual(got.shape, (70, 4, ieeg_fm.HIDDEN_DIM))
                np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6)
        finally:
            ieeg_fm.load_brainbert, rr.EXTRACT_CHUNK = saved
        self.assertEqual(rr.EXTRACT_CHUNK % rr.BB_BATCH, 0)
        self.assertEqual(rr.EXTRACT_CHUNK % rr.POPT_BATCH, 0)

    def test_sequences_round_trip(self):
        from experiments.run_ieeg_fm_regression import _chunk_sequences
        emb = np.random.RandomState(0).randn(130, 4).astype(np.float32)
        y = np.random.RandomState(1).randn(130, 5).astype(np.float32)
        X, Y, M = _chunk_sequences(emb, y, 64)
        self.assertEqual(X.shape, (3, 64, 4))
        self.assertEqual(int(M.sum()), 130)
        np.testing.assert_array_equal(X.reshape(-1, 4)[:130], emb)
        np.testing.assert_array_equal(Y.reshape(-1, 5)[:130], y)

    def test_only_the_unidirectional_head_is_causal(self):
        import torch
        from experiments.run_ieeg_fm_regression import TemporalHead
        torch.manual_seed(0)
        x = torch.randn(1, 16, 8)
        x2 = x.clone()
        x2[:, 10:] += 1.0                                         # change the future only
        for bi, causal in ((True, False), (False, True)):
            m = TemporalHead(8, 5, hidden=6, bidirectional=bi).eval()
            with torch.no_grad():
                same = torch.allclose(m(x)[:, :10], m(x2)[:, :10])
            self.assertEqual(same, causal)


if __name__ == "__main__":
    unittest.main(verbosity=2)
