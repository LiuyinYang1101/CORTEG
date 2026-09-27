"""CPU tests for the BrainTreebank intracranial-FM arms.

No third-party weights or BrainTreebank data are needed: the models are
replaced by small deterministic stand-ins, and everything else is the code the
paper numbers depend on -- BrainBERT's pooling, the event chunking, Brant's
window geometry and resampling, the cache keys, and the PopT LoRA injection.

Run:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import tracemalloc
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FM_SCRIPT = REPO / "scripts" / "table3_ieeg_fm_braintreebank.sh"
RUNNERS = {
    "experiments.run_ieeg_fm_baselines": REPO / "experiments" / "run_ieeg_fm_baselines.py",
    "experiments.run_popt_finetune_btb": REPO / "experiments" / "run_popt_finetune_btb.py",
}

try:
    import torch
except ImportError:                                   # pragma: no cover
    torch = None


class _StubBrainBERT:
    """Deterministic stand-in for BrainBERT: (b, T, 40) -> (b, T, 768).

    The output is a numpy array wrapped by torch.from_numpy, so its memory is
    visible to tracemalloc, which is what the memory test measures.
    """

    def __init__(self):
        self.W = np.random.RandomState(0).randn(40, 768).astype(np.float32) / 7.0
        self.batch_sizes = []

    def forward(self, chunk, mask, intermediate_rep=True):
        self.batch_sizes.append(int(chunk.shape[0]))
        out = np.tanh(chunk.numpy() @ self.W).astype(np.float32)
        return torch.from_numpy(out)


def _old_brainbert_embeddings(x_raw, fs, xyz_mm, reref, pool, batch_size, model):
    """The released implementation before per-batch pooling, kept as the reference:
    build every spectrogram, run every batch, concatenate, then pool."""
    import ieeg_fm
    N, C, _ = x_raw.shape
    specs = np.stack([ieeg_fm.preprocess_window(x_raw[n], fs, xyz_mm, reref)
                      for n in range(N)], axis=0)
    T_frames = specs.shape[2]
    flat = specs.reshape(N * C, T_frames, ieeg_fm.INPUT_DIM)
    outs = []
    for i in range(0, flat.shape[0], batch_size):
        chunk = torch.from_numpy(flat[i:i + batch_size]).float()
        mask = torch.zeros(chunk.shape[:2], dtype=torch.bool)
        with torch.no_grad():
            rep = model.forward(chunk, mask, intermediate_rep=True)
        outs.append(rep.detach().cpu())
    rep = torch.cat(outs, dim=0).numpy()
    if pool == "default":
        mid = T_frames // 2
        lo, hi = max(0, mid - ieeg_fm.POOL_HALF), mid + ieeg_fm.POOL_HALF
        return rep[:, lo:hi].mean(axis=1).reshape(N, C, ieeg_fm.HIDDEN_DIM)
    if pool == "mean":
        return rep.mean(axis=1).reshape(N, C, ieeg_fm.HIDDEN_DIM)
    return rep.reshape(N, C, T_frames, ieeg_fm.HIDDEN_DIM)


@unittest.skipIf(torch is None, "torch is not installed")
class TestBrainBERTPooling(unittest.TestCase):
    """Per-batch pooling must be bit-identical to pooling after concatenation."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(1)
        cls.fs = 2048.0
        cls.x = rng.randn(6, 5, int(1.5 * cls.fs)).astype(np.float32)   # Task A window
        cls.xyz = rng.randn(5, 3) * 10.0

    def _both(self, pool, batch_size, x=None):
        import ieeg_fm
        x = self.x if x is None else x
        new = ieeg_fm.brainbert_embeddings(x, fs=self.fs, xyz_mm=self.xyz, device="cpu",
                                           pool=pool, batch_size=batch_size,
                                           model=_StubBrainBERT())
        old = _old_brainbert_embeddings(x, self.fs, self.xyz, "laplacian_xyz", pool,
                                        batch_size, _StubBrainBERT())
        return new, old

    def test_default_pool_is_bit_identical(self):
        # 7 does not divide N*C = 30, so batches straddle windows, as they do
        # in real runs.
        new, old = self._both("default", 7)
        self.assertEqual(new.shape, (6, 5, 768))
        self.assertTrue(np.array_equal(new, old), f"max|d|={np.abs(new - old).max()}")

    def test_mean_and_none_pools_are_bit_identical(self):
        for pool in ("mean", "none"):
            new, old = self._both(pool, 4)
            self.assertEqual(new.shape, old.shape)
            self.assertTrue(np.array_equal(new, old), pool)

    def test_batches_never_exceed_batch_size(self):
        import ieeg_fm
        stub = _StubBrainBERT()
        ieeg_fm.brainbert_embeddings(self.x, fs=self.fs, xyz_mm=self.xyz, device="cpu",
                                     batch_size=7, model=stub)
        self.assertEqual(stub.batch_sizes, [7, 7, 7, 7, 2])

    def test_event_chunks_concatenate_to_the_unchunked_result(self):
        """--event_chunk splits events across calls; the result must not change."""
        import ieeg_fm
        kw = dict(fs=self.fs, xyz_mm=self.xyz, device="cpu", batch_size=7)
        whole = ieeg_fm.brainbert_embeddings(self.x, model=_StubBrainBERT(), **kw)
        parts = [ieeg_fm.brainbert_embeddings(self.x[a:b], model=_StubBrainBERT(), **kw)
                 for a, b in ((0, 2), (2, 5), (5, 6))]
        self.assertTrue(np.array_equal(whole, np.concatenate(parts, axis=0)))

    def test_no_unpooled_buffer_is_held(self):
        """Peak memory must scale with one batch, not with every frame of every window.

        The reference implementation is measured too, so the test proves it can
        tell the two apart.
        """
        import ieeg_fm
        x = np.random.RandomState(2).randn(48, 5, int(1.5 * self.fs)).astype(np.float32)
        full = 48 * 5 * 43 * 768 * 4          # every unpooled frame, float32 (~32 MB)

        def peak(fn):
            tracemalloc.start()
            try:
                fn()
                return tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()

        new = peak(lambda: ieeg_fm.brainbert_embeddings(
            x, fs=self.fs, xyz_mm=self.xyz, device="cpu", batch_size=8,
            model=_StubBrainBERT()))
        old = peak(lambda: _old_brainbert_embeddings(
            x, self.fs, self.xyz, "laplacian_xyz", "default", 8, _StubBrainBERT()))
        self.assertGreater(old, full, "the measurement cannot see the unpooled buffer")
        self.assertLess(new, full / 4, f"peak {new} bytes: an unpooled buffer is held")


class TestBrantGeometry(unittest.TestCase):
    """Brant reads the 6 s ending at the task window's right edge."""

    def test_windows_end_where_the_task_windows_end(self):
        from experiments.run_ieeg_fm_baselines import (
            BRANT_WINDOW, ENDPOINTS, window_for)
        self.assertEqual(BRANT_WINDOW["sentence_onset"], {"win_sec": 6.0, "pre_sec": -4.5})
        self.assertEqual(BRANT_WINDOW["word_nonword"], {"win_sec": 6.0, "pre_sec": -3.5})
        for ep, w in ENDPOINTS.items():
            b = window_for("brant", ep)
            self.assertEqual(b["pre_sec"] + b["win_sec"], w["pre_sec"] + w["win_sec"], ep)
            self.assertEqual(window_for("brainbert", ep), w)
            self.assertEqual(window_for("popt", ep), w)

    def test_footprint_fits_the_embargo(self):
        from data.braintreebank import EMBARGO_SEC
        from experiments.run_ieeg_fm_baselines import footprint_for
        for ep in ("sentence_onset", "word_nonword"):
            self.assertAlmostEqual(footprint_for("brant", ep), 6.11)
            self.assertLess(footprint_for("brant", ep), EMBARGO_SEC)
        self.assertEqual(footprint_for("brainbert", "word_nonword"), 5.0)
        self.assertEqual(footprint_for("popt", "sentence_onset"), 1.5)

    def test_patch_is_the_six_seconds_before_the_right_edge(self):
        import ieeg_fm
        fs = 2000.0
        stream = np.arange(4000, dtype=np.float32)[:, None].repeat(2, axis=1)   # 16 s
        center = 10000                                   # t = 5 s, native samples
        right = center + int(round((-2.5 + 5.0) * fs))   # Task B window ends at t+2.5
        end = int(ieeg_fm.brant_right_edges_250(np.array([right]), fs)[0])
        patch, padded = ieeg_fm._context_patches(stream, end, 1)
        self.assertFalse(padded)
        self.assertEqual(patch.shape, (2, 1, 1500))
        self.assertEqual(patch[0, 0, 0], (5.0 - 3.5) * 250)       # starts at t-3.5
        self.assertEqual(patch[0, 0, -1], (5.0 + 2.5) * 250 - 1)  # ends just before t+2.5

    def test_short_history_is_filled_with_real_signal(self):
        import ieeg_fm
        stream = np.arange(1, 1001, dtype=np.float32)[:, None]
        patch, padded = ieeg_fm._context_patches(stream, 1000, 1)
        self.assertTrue(padded)
        self.assertFalse((patch == 0).any(), "a short history must never be zero-padded")


class TestBrantResample(unittest.TestCase):

    def test_ratio_uses_the_measured_rate(self):
        import ieeg_fm
        # sub_3's measured rate: rounding it first would give 5/41 (250.006 Hz).
        self.assertEqual(ieeg_fm.brant_resample_ratio(2049.9480020451124), (976000, 8002997))
        self.assertEqual(ieeg_fm.brant_resample_ratio(2048.0), (125, 1024))

    def test_stream_matches_resample_poly_bit_for_bit(self):
        """The shared FIR must reproduce resample_poly's own design exactly."""
        import ieeg_fm
        from scipy.signal import resample_poly
        fs = 2048.5                                       # ratio 500/4097
        sig = np.random.RandomState(3).randn(3, 40000)
        out = ieeg_fm.brant_stream_at_250(lambda i: sig[i], 3, sig.shape[1], fs)
        up, down = ieeg_fm.brant_resample_ratio(fs)
        self.assertEqual(out.shape, (int(np.ceil(40000 * up / down)), 3))
        for i in range(3):
            ref = resample_poly(sig[i], up, down).astype(np.float32)
            self.assertTrue(np.array_equal(out[:, i], ref[:out.shape[0]]), i)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_stream_embeddings_keep_the_channel_axis(self):
        import ieeg_fm

        def et(mask, data, power, need_mask):            # (B,C,L,1500) -> (B*C,L,2048)
            b, c, l, _ = data.shape
            feat = torch.cat([power, data[..., :8]], dim=-1).reshape(b * c, l, 16)
            return feat.repeat(1, 1, 128)

        def ec(tz):
            return tz * 2.0, None

        stream = np.random.RandomState(4).randn(5000, 3).astype(np.float32)
        emb = ieeg_fm.brant_stream_embeddings(stream, [1600, 3000, 4999, 700],
                                              device="cpu", batch_size=3, model=(et, ec))
        self.assertEqual(emb.shape, (4, 3, 2048))
        self.assertTrue(np.isfinite(emb).all())
        self.assertFalse(np.allclose(emb[:, 0], emb[:, 1]), "channels were pooled")


class TestFmRunner(unittest.TestCase):

    def test_cache_key(self):
        """The event draw and the window key the cache; the probe seed does not."""
        from experiments.run_ieeg_fm_baselines import cache_path
        a = SimpleNamespace(fm="brant", endpoint="word_nonword", max_per_class=900,
                            event_seed=42, seed=42)
        name = os.path.basename(cache_path("sub_3", "trial000", a))
        self.assertEqual(name, "fm_brant_word_nonword_sub_3_trial000_win6.0_pre-3.5_n900_s42.npz")
        self.assertEqual(cache_path("sub_3", "trial000", SimpleNamespace(**{**vars(a), "seed": 1})),
                         cache_path("sub_3", "trial000", a))
        for change in ({"event_seed": 1}, {"max_per_class": 10}, {"endpoint": "sentence_onset"}):
            self.assertNotEqual(
                cache_path("sub_3", "trial000", SimpleNamespace(**{**vars(a), **change})),
                cache_path("sub_3", "trial000", a), change)

    def test_event_selection_uses_the_event_seed(self):
        src = RUNNERS["experiments.run_ieeg_fm_baselines"].read_text(encoding="utf-8")
        body = src[src.index("def _events("):src.index("def _device(")]
        self.assertEqual(body.count("args.event_seed"), 2)
        self.assertNotIn("args.seed", body)

    def test_popt_rejects_per_electrode_arms(self):
        from experiments import run_ieeg_fm_baselines as r
        argv = ["run_ieeg_fm_baselines", "--fm", "popt", "--arm", "single_elec_max"]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(r, "load_subject", side_effect=AssertionError("ran")), \
                open(os.devnull, "w") as devnull, mock.patch("sys.stderr", devnull):
            with self.assertRaises(SystemExit):
                r.main()

    def test_per_electrode_arm_on_population_embeddings_is_a_clear_error(self):
        from experiments.run_ieeg_fm_baselines import score_subject
        with self.assertRaises(ValueError):
            score_subject(np.zeros((40, 512)), np.zeros(40), [], "single_elec_max", 42)

    def test_parallel_probes_change_nothing(self):
        from experiments.run_ieeg_fm_baselines import score_subject
        rng = np.random.RandomState(5)
        y = np.tile([0, 1], 60)
        emb = rng.randn(120, 3, 6) + y[:, None, None] * np.array([0.0, 0.5, 1.0])[None, :, None]
        folds = [(np.arange(0, 60), None, np.arange(60, 120))]
        serial = score_subject(emb, y, folds, "single_elec_mean", 42)
        parallel = score_subject(emb, y, folds, "single_elec_mean", 42, n_jobs=2)
        self.assertEqual(serial[0], parallel[0])
        self.assertTrue(np.array_equal(serial[1], parallel[1]))

    def test_oracle_caveat_states_a_lower_bound(self):
        src = RUNNERS["experiments.run_ieeg_fm_baselines"].read_text(encoding="utf-8")
        caveat = src[src.index('out["caveat"]'):]
        caveat = caveat[:caveat.index(")\n")]
        for needed in ("0.53", "lower", "single_elec_mean"):
            self.assertIn(needed, caveat)


class TestPoptWeights(unittest.TestCase):

    def setUp(self):
        self._saved = os.environ.pop("POPT_WEIGHTS", None)

    def tearDown(self):
        os.environ.pop("POPT_WEIGHTS", None)
        if self._saved is not None:
            os.environ["POPT_WEIGHTS"] = self._saved

    def test_unset_falls_back_to_the_download(self):
        import ieeg_fm
        self.assertIsNone(ieeg_fm.popt_weights())

    def test_set_but_missing_is_an_error(self):
        import ieeg_fm
        os.environ["POPT_WEIGHTS"] = str(REPO / "no_such_checkpoint.pth")
        with self.assertRaises(SystemExit):
            ieeg_fm.popt_weights()

    def test_set_is_used(self):
        import ieeg_fm
        os.environ["POPT_WEIGHTS"] = str(Path(__file__))
        self.assertEqual(ieeg_fm.popt_weights(), str(Path(__file__)))

    def test_module_imports_without_torch(self):
        code = "import sys; sys.modules['torch'] = None; import ieeg_fm; ieeg_fm.get_emb"
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


@unittest.skipIf(torch is None, "torch is not installed")
class TestPoptLora(unittest.TestCase):
    """The weight-space LoRA must reach all four roles, in train() and in eval()."""

    def test_all_four_roles_are_adapted_and_receive_gradient(self):
        import torch.nn as nn
        from experiments.run_popt_finetune_btb import inject_popt_lora
        torch.manual_seed(0)
        enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(16, 2, 32, dropout=0.0, batch_first=True), 3,
            enable_nested_tensor=False)
        for p in enc.parameters():
            p.requires_grad = False
        n, per_block, roles = inject_popt_lora(SimpleNamespace(transformer_encoder=enc),
                                               n_last=2, r=4, alpha=16, dropout=0.2)
        self.assertEqual(roles, ["fc1", "fc2", "proj", "qkv"])
        # qkv (48x16), proj (16x16), fc1 (32x16), fc2 (16x32), each r*(d_in+d_out)
        per = 4 * (16 + 48) + 4 * (16 + 16) + 4 * (16 + 32) + 4 * (32 + 16)
        self.assertEqual(per_block, {1: per, 2: per})
        self.assertEqual(n, 2 * per)

        enc.train()
        enc(torch.randn(3, 5, 16)).pow(2).sum().backward()
        for bi in (1, 2):
            blk = enc.layers[bi]
            for mod, attr in ((blk.self_attn, "in_proj_weight"),
                              (blk.self_attn.out_proj, "weight"),
                              (blk.linear1, "weight"), (blk.linear2, "weight")):
                B = mod.parametrizations[attr][0].B
                self.assertIsNotNone(B.grad, f"block {bi} {attr}: no gradient")
                self.assertGreater(float(B.grad.abs().sum()), 0.0, f"block {bi} {attr}")
        self.assertFalse(hasattr(enc.layers[0].linear1, "parametrizations"))

    def test_eval_uses_the_adapted_weights(self):
        """The fused eval fast path must not bypass the adapters."""
        import torch.nn as nn
        from experiments.run_popt_finetune_btb import inject_popt_lora
        torch.manual_seed(0)
        enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(16, 2, 32, dropout=0.0, batch_first=True), 2,
            enable_nested_tensor=False)
        for p in enc.parameters():
            p.requires_grad = False
        inject_popt_lora(SimpleNamespace(transformer_encoder=enc), 2, 4, 16, 0.0)
        with torch.no_grad():
            for blk in enc.layers:
                for mod, attr in ((blk.self_attn, "in_proj_weight"),
                                  (blk.self_attn.out_proj, "weight"),
                                  (blk.linear1, "weight"), (blk.linear2, "weight")):
                    mod.parametrizations[attr][0].B.normal_()
        x = torch.randn(3, 5, 16)
        enc.train()
        with torch.no_grad():
            y_train = enc(x)
        enc.eval()
        with torch.no_grad():
            y_eval = enc(x)
        self.assertTrue(torch.allclose(y_train, y_eval, atol=1e-5))

    def test_standardiser_matches_standard_scaler(self):
        """Fit-set standardisation, up to the variance estimator: torch.std is
        unbiased, StandardScaler uses ddof=0, a factor sqrt(n/(n-1))."""
        from sklearn.preprocessing import StandardScaler
        from experiments.run_popt_finetune_btb import StandardizeAffine
        x = np.random.RandomState(6).randn(50, 8).astype(np.float32) * 3 + 1
        x[:, 3] = 2.0                                     # a zero-variance feature
        norm = StandardizeAffine(8)
        norm.fit(torch.from_numpy(x))
        got = norm(torch.from_numpy(x)).detach().numpy()
        ref = StandardScaler().fit_transform(x) * np.sqrt((len(x) - 1) / len(x))
        self.assertTrue(np.allclose(got, ref, atol=1e-5))
        self.assertTrue(np.allclose(got[:, 3], 0.0))


def _runner_flags(path: Path) -> dict:
    """{'--flag': (choices or None, takes_several_values)} from a runner's add_argument calls."""
    src = path.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r'add_argument\(\s*"(--[A-Za-z0-9_]+)"(.*?)\)\n', src, re.S):
        ch = re.search(r"choices=\[(.*?)\]", m.group(2), re.S)
        out[m.group(1)] = (set(re.findall(r'"([^"]*)"', ch.group(1))) if ch else None,
                           'nargs="+"' in m.group(2))
    return out


def _script_calls(text: str):
    """[(runner, [argv, ...])] for each runner call in the FM script, in file order.

    Every `for VAR in a b c` enclosing a call is expanded, so a loop value that
    reaches a runner is checked like a literal one.
    """
    text = text.replace("\\\n", " ")
    loops, calls = [], []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r"for (\w+) in (.*?); do$", s)
        if m:
            loops.append((m.group(1), m.group(2).split()))
            continue
        if s == "done":
            loops.pop()
            continue
        if "python -m " not in s:
            continue
        toks = shlex.split(s, comments=True)
        module, argv = toks[toks.index("-m") + 1], toks[toks.index("-m") + 2:]
        expanded = [argv]
        for var, values in loops:
            expanded = [[t.replace(f"${var}", v).replace(f"${{{var}}}", v) for t in a]
                        for a in expanded for v in values]
        calls.append((module, expanded))
    return calls


def _flag_values(argv, flag):
    """The values given to `flag` in argv (every token up to the next flag)."""
    if flag not in argv:
        return None
    rest = argv[argv.index(flag) + 1:]
    vals = []
    for t in rest:
        if t.startswith("--"):
            break
        vals.append(t)
    return vals


class TestFmScript(unittest.TestCase):
    """Each runner call in the FM script is checked against the runner it calls."""

    @classmethod
    def setUpClass(cls):
        cls.calls = _script_calls(FM_SCRIPT.read_text(encoding="utf-8"))

    def _argvs(self, module=None):
        return [(m, a) for m, argvs in self.calls if module in (None, m) for a in argvs]

    def test_every_flag_is_known_and_valid(self):
        problems = []
        self.assertTrue(self.calls, "no runner call found")
        for module, argv in self._argvs():
            self.assertIn(module, RUNNERS, f"unexpected runner {module}")
            spec = _runner_flags(RUNNERS[module])
            for tok in argv:
                if not tok.startswith("--"):
                    continue
                if tok not in spec:
                    problems.append(f"{module}: unknown flag {tok}")
                    continue
                choices, several = spec[tok]
                vals = _flag_values(argv, tok)
                if len(vals) > 1 and not several:
                    problems.append(f"{module}: {tok} takes one value, got {vals}")
                for val in vals:
                    if "$" in val and not re.fullmatch(r'\$\{?(SEED|N_JOBS|ROOT)\}?', val):
                        problems.append(f"{module}: {tok}={val!r} is not expanded")
                    elif choices and val not in choices:
                        problems.append(f"{module}: {tok}={val!r} not in {sorted(choices)}")
        self.assertEqual(problems, [], "\n" + "\n".join(problems))

    def test_every_paper_arm_is_run(self):
        base = "experiments.run_ieeg_fm_baselines"
        probes = {(_flag_values(a, "--fm")[0], _flag_values(a, "--endpoint")[0]):
                  set(_flag_values(a, "--arm") or []) for m, a in self._argvs(base)
                  if "--embed_only" not in a}
        for fm in ("brainbert", "brant"):
            for ep in ("sentence_onset", "word_nonword"):
                self.assertEqual(probes.get((fm, ep)),
                                 {"single_elec_max", "single_elec_mean", "pop_meanpool"},
                                 (fm, ep))
        for ep in ("sentence_onset", "word_nonword"):
            self.assertEqual(probes.get(("popt", ep)), {"pop_meanpool"}, ep)
        trained = {(_flag_values(a, "--endpoint")[0], _flag_values(a, "--mode")[0])
                   for m, a in self._argvs("experiments.run_popt_finetune_btb")}
        self.assertEqual(trained, {(ep, mode) for ep in ("sentence_onset", "word_nonword")
                                   for mode in ("lora", "full_ft", "head_only")})

    def test_order_and_stages(self):
        """BrainBERT before PopT (PopT reads its cache); PopT before Brant (so a
        Brant failure cannot cost the PopT rows); every per-electrode probe from
        the caches, on CPU."""
        def kind(module, argv):
            if module.endswith("run_popt_finetune_btb"):
                return "popt_trained"
            fm = _flag_values(argv, "--fm")[0]
            return (f"{fm}_embed" if "--embed_only" in argv else
                    "popt_frozen" if fm == "popt" else f"{fm}_probe")
        order = []
        for module, argv in self._argvs():
            k = kind(module, argv)
            if k not in order:
                order.append(k)
        self.assertEqual(order, ["brainbert_embed", "popt_frozen", "popt_trained",
                                 "brant_embed", "brainbert_probe", "brant_probe"])
        for module, argv in self._argvs("experiments.run_ieeg_fm_baselines"):
            if set(_flag_values(argv, "--arm") or []) & {"single_elec_max",
                                                          "single_elec_mean"}:
                self.assertIn("--from_cache", argv)
                self.assertEqual(_flag_values(argv, "--device"), ["cpu"])
                self.assertEqual(_flag_values(argv, "--n_jobs"), ["$N_JOBS"])


class TestFmScriptRuns(unittest.TestCase):
    """Run the script with a stand-in `python` that records each call."""

    def _run(self, stage=None, fail_on=None, n_jobs=None):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "calls.log"
            shim = Path(d) / "python"
            shim.write_text("#!/usr/bin/env bash\n"
                            f'echo "$*" >> "{log}"\n'
                            + (f'[[ "$*" == *"{fail_on}"* ]] && exit 3\n' if fail_on else "")
                            + "exit 0\n", encoding="utf-8")
            shim.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if k not in ("STAGE", "N_JOBS")}
            env.update(PATH=f"{d}{os.pathsep}{env.get('PATH', '')}",
                       CORTEG_OUTPUT_ROOT=d)
            if stage is not None:
                env["STAGE"] = stage
            if n_jobs is not None:
                env["N_JOBS"] = n_jobs
            r = subprocess.run(["bash", str(FM_SCRIPT)], cwd=REPO, env=env,
                               capture_output=True, text=True, timeout=60)
            calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        return r, calls

    def test_a_failed_call_does_not_stop_the_others(self):
        r, calls = self._run(fail_on="--fm brainbert --endpoint sentence_onset --embed_only")
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertEqual(len(calls), 2 + 2 + 6 + 2 + 4, calls)   # every call still ran
        self.assertIn("1 call(s) failed", r.stderr)
        self.assertIn("--fm brainbert --endpoint sentence_onset --embed_only", r.stderr)

    def test_all_succeed(self):
        r, calls = self._run()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(calls), 16)

    def test_stages_split_the_calls(self):
        r, gpu = self._run(stage="gpu")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(gpu), 12)
        self.assertFalse([c for c in gpu if "--from_cache" in c])
        r, probe = self._run(stage="probe")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(probe), 4)
        self.assertTrue(all("--from_cache" in c for c in probe), probe)
        r, _ = self._run(stage="bogus")
        self.assertEqual(r.returncode, 2)

    def test_n_jobs_default_is_the_paper_runs(self):
        _, calls = self._run(stage="probe")
        want = str(min(12, max(1, (os.cpu_count() or 1) - 4)))
        for c in calls:
            self.assertIn(f"--n_jobs {want} ", c + " ")
        _, calls = self._run(stage="probe", n_jobs="3")
        self.assertTrue(all("--n_jobs 3 " in c for c in calls))


class TestFmRunnerStages(unittest.TestCase):
    """Multi-arm scoring, --embed_only / --from_cache, and the result files."""

    @staticmethod
    def _data(n=120, c=3, d=6, seed=5):
        rng = np.random.RandomState(seed)
        y = np.tile([0, 1], n // 2)
        emb = (rng.randn(n, c, d) + y[:, None, None]
               * np.linspace(0.0, 1.0, c)[None, :, None]).astype(np.float32)
        ev = np.arange(n, dtype=np.float64) * 20.0          # 20 s apart: folds exist
        return {"emb": emb, "y": y, "event_times": ev,
                "electrodes": [f"E{i}" for i in range(c)], "fs": 2048.0,
                "trial": "trial000"}

    def test_per_electrode_probes_are_fitted_once(self):
        from experiments import run_ieeg_fm_baselines as r
        d = self._data()
        folds = [(np.arange(0, 60), None, np.arange(60, 120)),
                 (np.arange(0, 80), None, np.arange(80, 120))]
        real = r.probe_auroc
        with mock.patch.object(r, "probe_auroc", side_effect=real) as spy:
            both = r.score_arms(d["emb"], d["y"], folds,
                                ["single_elec_max", "single_elec_mean", "pop_meanpool"], 42)
        self.assertEqual(spy.call_count, 3 * 2 + 2)          # C x folds, + the pooled probe
        for arm in ("single_elec_max", "single_elec_mean", "pop_meanpool"):
            alone = r.score_subject(d["emb"], d["y"], folds, arm, 42)
            self.assertEqual(both[arm][0], alone[0], arm)
        self.assertEqual(both["single_elec_max"][0], float(np.nanmax(both["single_elec_max"][1])))
        self.assertEqual(both["single_elec_mean"][0],
                         float(np.nanmean(both["single_elec_mean"][1])))

    def test_result_names(self):
        import argparse
        from experiments.run_ieeg_fm_baselines import result_name, result_tags
        p = argparse.ArgumentParser()
        p.add_argument("--subjects", nargs="+", default=[f"sub_{i}" for i in range(1, 11)])
        p.add_argument("--max_per_class", type=int, default=900)
        p.add_argument("--event_seed", type=int, default=42)
        p.add_argument("--n_folds", type=int, default=4)
        p.add_argument("--trial", default=None)
        p.add_argument("--n_jobs", type=int, default=1)
        p.add_argument("--no_amp", dest="use_amp", action="store_false", default=True)
        self.assertEqual(result_tags(p.parse_args([]), p), [])
        self.assertEqual(result_tags(p.parse_args(["--n_jobs", "12"]), p), [])
        full = p.parse_args(["--subjects"] + [f"sub_{i}" for i in range(10, 0, -1)])
        self.assertEqual(result_tags(full, p), [], "subject order is not a setting")
        smoke = p.parse_args(["--subjects", "sub_10", "sub_9", "--max_per_class", "50",
                              "--event_seed", "7", "--n_folds", "2", "--trial", "trial001",
                              "--no_amp"])
        self.assertEqual(result_tags(smoke, p),
                         ["es7", "n50", "n_folds2", "sub_9+sub_10", "trial001", "no_use_amp"])
        # The fine-tune runner's --batch_size is the training batch: it is tagged.
        from experiments.run_popt_finetune_btb import NAME_NEUTRAL_FT
        q = argparse.ArgumentParser()
        q.add_argument("--batch_size", type=int, default=32)
        q.add_argument("--eval_batch_size", type=int, default=64)
        q.add_argument("--epochs", type=int, default=60)
        ft = q.parse_args(["--batch_size", "16", "--eval_batch_size", "8", "--epochs", "1"])
        self.assertEqual(result_tags(ft, q, NAME_NEUTRAL_FT), ["batch_size16", "epochs1"])
        self.assertEqual(result_tags(ft, q), ["epochs1"], "the frozen runner's is not")
        self.assertEqual(result_name("brant_word_nonword_single_elec_max", 42, []),
                         "brant_word_nonword_single_elec_max_seed42.json")
        self.assertEqual(result_name("popt_sentence_onset_lora", 1, ["n50", "sub_9"]),
                         "popt_sentence_onset_lora_seed1_n50_sub_9.json")

    def _main(self, argv, tmp, load=None):
        from experiments import run_ieeg_fm_baselines as r
        with mock.patch.object(sys, "argv", ["run_ieeg_fm_baselines"] + argv), \
                mock.patch.object(r, "load_subject", side_effect=load or
                                  (lambda s, a: self._data())), \
                mock.patch.object(r, "trial_of", return_value="trial000"), \
                mock.patch.object(r, "cache_path",
                                  return_value=os.path.join(tmp, "no_such_cache.npz")), \
                open(os.devnull, "w") as devnull, mock.patch("sys.stdout", devnull):
            r.main()
        return sorted(os.listdir(tmp))

    def test_one_call_writes_one_file_per_arm(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            files = self._main(["--fm", "brant", "--arm", "single_elec_max",
                                "single_elec_mean", "pop_meanpool", "--subjects", "sub_9",
                                "--n_folds", "2", "--save_root", tmp], tmp)
            self.assertEqual(files, [f"brant_sentence_onset_{a}_seed42_n_folds2_sub_9.json"
                                     for a in ("pop_meanpool", "single_elec_max",
                                               "single_elec_mean")])
            outs = {f: json.loads(Path(tmp, f).read_text(encoding="utf-8")) for f in files}
        for f, out in outs.items():
            arm = out["arm"]
            self.assertIn(f"_{arm}_", f)
            self.assertEqual(out["args"]["arm"], arm)
            self.assertIsNone(out["cohort_sd"], "one subject has no SD")
            self.assertEqual("caveat" in out, arm == "single_elec_max")
        mx = outs[files[1]]["per_subject"]["sub_9"]
        mn = outs[files[2]]["per_subject"]["sub_9"]
        self.assertEqual(mx["per_electrode_auroc"], mn["per_electrode_auroc"])
        self.assertEqual(mx["auroc"], max(mx["per_electrode_auroc"]))

    def test_embed_only_writes_no_result(self):
        import tempfile
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            files = self._main(["--fm", "brainbert", "--embed_only", "--subjects", "sub_9",
                                "sub_10", "--save_root", tmp], tmp,
                               load=lambda s, a: seen.append(s) or self._data())
        self.assertEqual(files, [])
        self.assertEqual(seen, ["sub_9", "sub_10"])

    def test_from_cache_never_builds(self):
        import tempfile
        from experiments import run_ieeg_fm_baselines as r
        args = SimpleNamespace(fm="brant", endpoint="word_nonword", trial="trial000",
                               max_per_class=900, event_seed=42, no_cache=False,
                               from_cache=True)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(r, "cache_path", return_value=os.path.join(tmp, "x.npz")), \
                mock.patch.object(r, "movie_of", side_effect=AssertionError("built")):
            with self.assertRaises(FileNotFoundError):
                r.load_subject("sub_3", args)
        for bad in (["--from_cache", "--no_cache"], ["--from_cache", "--embed_only"]):
            with mock.patch.object(sys, "argv", ["x", "--fm", "brant"] + bad), \
                    open(os.devnull, "w") as devnull, mock.patch("sys.stderr", devnull):
                with self.assertRaises(SystemExit, msg=bad):
                    r.main()

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_popt_builds_the_shared_brainbert_cache_at_its_own_batch(self):
        """--batch_size must not reach the BrainBERT cache every PopT arm shares."""
        import tempfile
        from experiments import run_ieeg_fm_baselines as r
        real = r.load_subject
        nested = []

        def load(subj, a):
            if a.fm == "brainbert":
                nested.append(a.batch_size)
                d = self._data(n=4, c=2, d=768)
                d["electrodes"] = ["E0", "E1"]
                return d
            return real(subj, a)

        popt_batches = []

        def popt_embeddings(x, lip, **kw):
            popt_batches.append(kw["batch_size"])
            return np.zeros((len(kw["per_electrode_embeddings"]), 512), np.float32)

        args = SimpleNamespace(fm="popt", endpoint="sentence_onset", trial="trial000",
                               max_per_class=900, event_seed=42, event_chunk=200,
                               batch_size=8, device="cpu", no_cache=False)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(r, "load_subject", side_effect=load), \
                mock.patch.object(r, "cache_path", return_value=os.path.join(tmp, "p.npz")), \
                mock.patch.object(r, "movie_of", return_value="m"), \
                mock.patch.object(r, "load_electrode_map", return_value=({"E0": 0, "E1": 1}, [])), \
                mock.patch.object(r, "load_localization",
                                  return_value={"E0": [1, 2, 3], "E1": [4, 5, 6]}), \
                mock.patch.object(r, "clean_electrodes", return_value=["E0", "E1"]), \
                mock.patch.object(r, "estimate_fs", return_value=2048.0), \
                mock.patch.object(r, "build_time_to_sample", return_value=None), \
                mock.patch.object(r, "_events", return_value=(np.zeros(4), np.zeros(4))), \
                mock.patch.object(ieeg_fm_module(), "load_popt_model", return_value=None), \
                mock.patch.object(ieeg_fm_module(), "popt_embeddings",
                                  side_effect=popt_embeddings), \
                open(os.devnull, "w") as devnull, mock.patch("sys.stdout", devnull):
            out = r.load_subject("sub_9", args)
        self.assertEqual(nested, [None])
        self.assertEqual(popt_batches, [r.POPT_BATCH])
        self.assertEqual(out["emb"].shape, (4, 512))
        self.assertNotIn("popt", r.DEFAULT_BATCH)


def ieeg_fm_module():
    import ieeg_fm
    return ieeg_fm


if __name__ == "__main__":
    unittest.main()
