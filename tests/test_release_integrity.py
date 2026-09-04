"""Release-integrity tests for the CORTEG public repo.

These guard the class of defect that made the v1 release reproduce the wrong
experiment: a shell script that silently disagrees with the runner it calls.
Everything here is static except the two tests in `TestPretrainedLoads`, which
build a model on CPU and skip cleanly when the ST-EEGFormer backbone is absent.

Run:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "experiments" / "run_regression_hilo_clean.py"
BTB_RUNNER = REPO / "experiments" / "run_btb_classification.py"
FM_RUNNER = REPO / "experiments" / "run_ieeg_fm_baselines.py"
ALL_SCRIPTS = sorted((REPO / "scripts").glob("*.sh"))
RUNNER_CALL = "python -m experiments.run_regression_hilo_clean"
BTB_RUNNER_CALL = "python -m experiments.run_btb_classification"
FM_RUNNER_CALL = "python -m experiments.run_ieeg_fm_baselines"

# Each script is validated against the runner it actually calls; checking a
# BrainTreebank script against the Stanford parser reports every BTB flag as
# unknown.
SCRIPTS = [s for s in ALL_SCRIPTS
           if RUNNER_CALL in s.read_text(encoding="utf-8")]
BTB_SCRIPTS = [s for s in ALL_SCRIPTS
               if BTB_RUNNER_CALL in s.read_text(encoding="utf-8")]
FM_SCRIPTS = [s for s in ALL_SCRIPTS
              if FM_RUNNER_CALL in s.read_text(encoding="utf-8")]

sys.path.insert(0, str(REPO))


# ─────────────────────────── argparse spec, parsed statically ───────────────
def _balanced(src: str, start: int) -> str:
    """Text of the call whose opening '(' is at `start`."""
    depth, i = 0, start
    while i < len(src):
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
        i += 1
    raise ValueError("unbalanced parentheses")


def parser_spec(path: Path) -> dict:
    """{'--flag': {'choices': set|None, 'store_true': bool}} parsed from source.

    The runner builds its parser inside main(), so it cannot be imported
    without executing the run; parse it instead.
    """
    src = path.read_text(encoding="utf-8")
    spec = {}
    for m in re.finditer(r'add_argument\(', src):
        call = _balanced(src, m.end() - 1)
        flag = re.search(r'"(--[A-Za-z0-9_]+)"', call)
        if not flag:
            continue
        ch = re.search(r"choices=\[(.*?)\]", call, re.S)
        spec[flag.group(1)] = {
            "choices": set(re.findall(r'"([^"]*)"', ch.group(1))) if ch else None,
            "store_true": 'action="store_true"' in call,
        }
    return spec


def script_flags(text: str):
    """Yield (flag, value|None) for every flag in a shell script.

    Handles line continuations, bash arrays and quoting. A flag followed by
    another flag (or by end of list) is treated as valueless.
    """
    text = text.replace("\\\n", " ")          # join continuations
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        line = line.replace("(", " ").replace(")", " ")
        try:
            toks = shlex.split(line, comments=True)
        except ValueError:
            toks = line.split()
        for i, tok in enumerate(toks):
            if not tok.startswith("--"):
                continue
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            # A value must be on the same logical line and not itself a flag.
            val = None if (nxt is None or nxt.startswith("--")) else nxt
            # A shell variable cannot be checked against `choices` statically;
            # report it as valueless rather than as an illegal literal.
            if val is not None and "$" in val:
                val = None
            yield tok, val


def _arrays(text: str):
    """Every `NAME=( ... )` bash array in `text`, as {name: body}."""
    return {m.group(1): m.group(2)
            for m in re.finditer(r"^(\w+)=\((.*?)\)", text, re.M | re.S)}


def _functions(text: str):
    """Every `name() { ... }` shell function in `text`, as {name: body}."""
    return {m.group(1): m.group(2)
            for m in re.finditer(r"^(\w+)\(\)\s*\{\n(.*?)^\}", text, re.M | re.S)}


def _invocations(text: str, runner_call: str = RUNNER_CALL):
    """Each runner invocation in `text`, with bash expansions resolved.

    Handles the direct `python -m experiments...` form and any wrapper function
    (`run`, `run_no_fuser`) that calls it, inlining the function body and every
    `"${ARRAY[@]}"` it expands. Resolving the expansion rather than grepping the
    file is what makes the per-call check sound: dropping a flag from a shared
    array then shows up on every call that used it.
    """
    arrays, funcs = _arrays(text), _functions(text)
    runners = {n for n, b in funcs.items() if runner_call in b}

    def expand(s, depth=0):
        if depth > 5:
            return s
        for name, body in arrays.items():
            s = s.replace('"${%s[@]}"' % name, body)
        return expand(s, depth + 1) if any(
            '"${%s[@]}"' % n in s for n in arrays) else s

    joined = text.replace("\\\n", " ")
    chunks, cur, started = [], [], None
    for line in joined.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        called = next((n for n in runners if stripped.startswith(n + " ")), None)
        starts = runner_call in line or called is not None
        if starts:
            if started is not None:
                chunks.append((started, "\n".join(cur)))
            cur, started = [line], called
        elif started is not None or cur:
            if not stripped and cur:
                chunks.append((started, "\n".join(cur))); cur, started = [], None
            elif cur:
                cur.append(line)
    if cur:
        chunks.append((started, "\n".join(cur)))

    out = []
    for fname, seg in chunks:
        body = funcs.get(fname, "") if fname else ""
        out.append((seg.strip().splitlines()[0] if seg.strip() else "", expand(seg + "\n" + body)))
    return out


class TestScriptsMatchRunner(unittest.TestCase):
    """Every flag a script passes must exist, and its value must be legal."""

    @classmethod
    def setUpClass(cls):
        cls.spec = parser_spec(RUNNER)

    def test_scripts_exist(self):
        self.assertTrue(SCRIPTS, "no shell scripts found")

    def test_every_flag_is_known_and_valid(self):
        """Every flag a script passes must exist, and its value must be legal.

        Catches the three ways a script silently diverges from the runner: an
        unknown flag (argparse would abort), a value outside `choices`, and a
        value handed to a `store_true` flag (which argparse reads as a
        positional, not as "off").
        """
        problems = []
        for sh in SCRIPTS:
            for flag, val in script_flags(sh.read_text(encoding="utf-8")):
                if flag not in self.spec:
                    problems.append(f"{sh.name}: unknown flag {flag}")
                    continue
                info = self.spec[flag]
                if info["store_true"] and val is not None:
                    problems.append(
                        f"{sh.name}: {flag} is store_true but got value {val!r} "
                        "(omit the flag to turn it off)")
                elif info["choices"] and val is not None and val not in info["choices"]:
                    problems.append(
                        f"{sh.name}: {flag}={val!r} not in {sorted(info['choices'])}")
        self.assertEqual(problems, [], "\n" + "\n".join(problems))

    def test_every_runner_call_loads_a_pretrained_backbone(self):
        """The v1 defect: no script passed --model_kwargs_json, so every run
        silently trained a randomly-initialised backbone."""
        problems = []
        for sh in SCRIPTS:
            text = sh.read_text(encoding="utf-8")
            if RUNNER_CALL not in text:
                continue
            for label, resolved in _invocations(text):
                # --no_pretrained exempts THIS call only, never the whole file:
                # one random-init row must not disable the check for its siblings.
                if "--model_kwargs_json" in resolved or "--no_pretrained" in resolved:
                    continue
                problems.append(f"{sh.name}: {label[:70]!r} loads no backbone")
        self.assertEqual(problems, [], "\n" + "\n".join(problems))

    def test_referenced_configs_exist(self):
        missing = []
        for sh in SCRIPTS:
            text = sh.read_text(encoding="utf-8")
            for cfg in re.findall(r'(configs/steegformer_[A-Za-z0-9_${}]+\.json)', text):
                for variant in (["small", "base", "large"] if "$" in cfg else [None]):
                    p = re.sub(r'\$\{?VARIANT\}?', variant, cfg) if variant else cfg
                    if not (REPO / p).exists():
                        missing.append(f"{sh.name}: {p}")
        self.assertEqual(missing, [], "\n" + "\n".join(missing))

    def test_loo_ft_stage2_actually_finetunes(self):
        """--finetune_from is only read under --train_mode finetune."""
        sh = REPO / "scripts" / "table1_corteg_loo_ft.sh"
        text = sh.read_text(encoding="utf-8")
        self.assertIn("--finetune_from", text)
        idx = text.index("--finetune_from")
        stage2 = text[max(0, idx - 900) : idx]
        self.assertIn("--train_mode finetune", stage2,
                      "Stage 2 passes --finetune_from without --train_mode finetune")


class TestPretrainedLoads(unittest.TestCase):
    """The v1 defect end to end: a config must actually change the weights.

    Skips cleanly when the ST-EEGFormer backbone is not present.
    """

    def test_pretrained_weights_load(self):
        import json as _json
        cfg = REPO / "configs" / "steegformer_small.json"
        rel = _json.loads(cfg.read_text(encoding="utf-8"))["pretrained"]["path"]
        import paths as paths_mod
        if not os.path.exists(paths_mod.resolve_pretrained_path(rel)):
            self.skipTest("ST-EEGFormer backbone not present; see README.md")

        import types
        import numpy as np
        from experiments.run_regression_hilo_clean import build_model

        # The shipped manifest, not a file under the author's output root: this
        # test has to run for anyone who clones the repo.
        base = json.loads((REPO / "checkpoints" / "corteg_stanford_pooled.json")
                          .read_text(encoding="utf-8"))["build_args"]
        xyz = np.random.RandomState(0).randn(64, 3).astype(np.float32) * 30.0

        def qkv(**over):
            a = dict(base); a.update(over)
            m = build_model(types.SimpleNamespace(**a), C_in=64, T_in=128,
                            ecog_xyz_m=xyz, d_out=5)
            return float(m.backbone.blocks[0].attn.qkv.weight.detach().std())

        rand = qkv(model_kwargs_json="", no_pretrained=True)
        pre = qkv(model_kwargs_json=str(cfg), no_pretrained=False)
        self.assertNotAlmostEqual(rand, pre, places=3,
                                  msg="config did not change the backbone weights")

    def test_missing_config_refuses(self):
        """No config and no --no_pretrained must stop, not silently randomise."""
        import types
        import numpy as np
        from experiments.run_regression_hilo_clean import build_model
        base = json.loads((REPO / "checkpoints" / "corteg_stanford_pooled.json")
                          .read_text(encoding="utf-8"))["build_args"]
        base.update(model_kwargs_json="", no_pretrained=False)
        xyz = np.random.RandomState(0).randn(64, 3).astype(np.float32) * 30.0
        with self.assertRaises(SystemExit):
            build_model(types.SimpleNamespace(**base), C_in=64, T_in=128,
                        ecog_xyz_m=xyz, d_out=5)


class TestReleasedCheckpoint(unittest.TestCase):
    """The released adapter must keep loading as the repo is cleaned up.

    This is the regression anchor for the strip-down: removing an argument or a
    merge strategy that the checkpoint's recorded build_args still name would
    break loading, and this test says so immediately.
    """

    @classmethod
    def setUpClass(cls):
        cls.ckpt = REPO / "checkpoints" / "corteg_stanford_pooled.pt"
        if not cls.ckpt.exists():
            raise unittest.SkipTest("released checkpoint not present")

    def test_manifest_matches_bytes_and_hash(self):
        from load_corteg import verify_checkpoint
        man = verify_checkpoint()
        self.assertEqual(man["trainable_params"], 297236)

    def test_paper_number_matches_manifest(self):
        """Manifest r must equal the README's CORTEG (pooled) finger score."""
        import re as _re
        man = json.loads((REPO / "checkpoints" / "corteg_stanford_pooled.json")
                         .read_text(encoding="utf-8"))
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        row = next((l for l in readme.splitlines()
                    if "CORTEG (pooled)" in l and "|" in l), None)
        self.assertIsNotNone(row, "README has no CORTEG (pooled) row")
        nums = _re.findall(r"0\.\d+", row)
        self.assertTrue(nums, f"no number in README row: {row!r}")
        self.assertEqual(float(nums[0]), round(man["paper"]["value"], 3),
                         f"README says {nums[0]}, manifest says {man['paper']['value']}")

    def test_checkpoint_rebuilds_and_loads(self):
        """Rebuild from the manifest's build_args and load every tensor."""
        import numpy as np
        from load_corteg import load_corteg, predict
        cfg = REPO / "configs" / "steegformer_small.json"
        import paths as paths_mod
        rel = json.loads(cfg.read_text(encoding="utf-8"))["pretrained"]["path"]
        if not os.path.exists(paths_mod.resolve_pretrained_path(rel)):
            self.skipTest("ST-EEGFormer backbone not present; see README.md")
        C, T_LO, T_HI = 46, 128, 200        # real Stanford dims, subject bp
        xyz = np.random.RandomState(0).randn(C, 3).astype(np.float32) * 0.03
        model = load_corteg(C_in=C, T_in=T_LO, ecog_xyz_mm=xyz, d_out=5)
        rs = np.random.RandomState(1)
        y = predict(model, rs.randn(3, C, T_LO), rs.randn(3, C, T_HI))
        self.assertEqual(y.shape, (3, 5))

    def test_hi_patch_embed_is_warm_started(self):
        """The hi patch embed must be interpolated from the pretrained lo weights.

        The scalp-EEG checkpoint has no patch_embed_hi, so the load reports it as
        a merely missing key and training proceeds from random init without any
        error. A strip-down once deleted this call while keeping the function,
        and inference-only checks could not see it -- hence this test.
        """
        import types
        import numpy as np
        import torch
        import paths as paths_mod
        cfg = REPO / "configs" / "steegformer_small.json"
        rel = json.loads(cfg.read_text(encoding="utf-8"))["pretrained"]["path"]
        if not os.path.exists(paths_mod.resolve_pretrained_path(rel)):
            self.skipTest("ST-EEGFormer backbone not present; see README.md")
        from experiments.run_regression_hilo_clean import build_model

        base = json.loads((REPO / "checkpoints" / "corteg_stanford_pooled.json")
                          .read_text(encoding="utf-8"))["build_args"]
        a = dict(base)
        a.update(model_kwargs_json=str(cfg), no_pretrained=False)
        xyz = np.random.RandomState(0).randn(46, 3).astype(np.float32) * 30.0
        m = build_model(types.SimpleNamespace(**a), C_in=46, T_in=128,
                        ecog_xyz_m=xyz, d_out=5)

        hi = m.backbone.patch_embed_hi.proj.weight.detach()
        lo = m.backbone.patch_embed.proj.weight.detach()
        want = torch.nn.functional.interpolate(
            lo.reshape(lo.shape[0], 1, -1), size=hi.shape[-1],
            mode="linear", align_corners=False).reshape(hi.shape)
        self.assertLess(float((hi - want).abs().max()), 1e-5,
                        "patch_embed_hi is not the interpolated lo embed — the "
                        "warm start is not being applied")

    def test_build_args_only_name_supported_flags(self):
        """Every arg recorded in the manifest must still exist on the runner.

        This is what catches an over-eager cleanup: delete --use_ea from the
        parser while the checkpoint's build_args still lists it and the release
        stops being self-describing.
        """
        spec = parser_spec(RUNNER)
        man = json.loads((REPO / "checkpoints" / "corteg_stanford_pooled.json")
                         .read_text(encoding="utf-8"))
        stale = sorted(k for k in man["build_args"] if f"--{k}" not in spec)
        self.assertEqual(stale, [],
                         "manifest build_args name flags the runner no longer has: "
                         f"{stale} — regenerate the manifest after the cleanup")


class TestBrainTreebank(unittest.TestCase):
    """The BrainTreebank arm: same model, different task and protocol."""

    def test_btb_script_flags_are_known_and_valid(self):
        spec = parser_spec(BTB_RUNNER)
        self.assertTrue(BTB_SCRIPTS, "no BrainTreebank script found")
        problems = []
        for sh in BTB_SCRIPTS:
          for flag, val in script_flags(sh.read_text(encoding="utf-8")):
            if flag not in spec:
                problems.append(f"{sh.name}: unknown flag {flag}")
                continue
            info = spec[flag]
            if info["store_true"] and val is not None:
                problems.append(f"{sh.name}: {flag} is store_true but got {val!r}")
            elif info["choices"] and val is not None and val not in info["choices"]:
                problems.append(f"{sh.name}: {flag}={val!r} not in {sorted(info['choices'])}")
        self.assertEqual(problems, [], "\n" + "\n".join(problems))

    def test_every_btb_call_loads_a_backbone(self):
        """Same defect as Stanford: a run without the config is the ablation."""
        problems = []
        for sh in BTB_SCRIPTS:
            for label, resolved in _invocations(sh.read_text(encoding="utf-8"),
                                                BTB_RUNNER_CALL):
                if ("--model_kwargs_json" not in resolved
                        and "--no_pretrained" not in resolved):
                    problems.append(f"{sh.name}: {label[:60]!r}")
        self.assertEqual(problems, [], f"calls with no backbone: {problems}")

    def test_knn_adapter_starts_from_the_knn_prior(self):
        """sigma=0 zeroes every Gaussian weight, so the adapter starts uniform.

        Checked as behaviour, not as a literal: build the model the BrainTreebank
        runner builds and confirm the spatial adapter's initial attention is not
        one constant across all EEG slots.
        """
        import types
        import numpy as np
        import torch
        import paths as paths_mod
        cfg = REPO / "configs" / "steegformer_small.json"
        rel = json.loads(cfg.read_text(encoding="utf-8"))["pretrained"]["path"]
        if not os.path.exists(paths_mod.resolve_pretrained_path(rel)):
            self.skipTest("ST-EEGFormer backbone not present; see README.md")
        from experiments.run_btb_classification import build_corteg

        args = types.SimpleNamespace(
            model_kwargs_json=str(cfg), no_pretrained=False,
            steegformer_variant="small", merge_strategy="layerwise_gate",
            layerwise_gate_bottleneck=16, layerwise_gate_act="tanh",
            head_dropout=0.1, lora_r=4, lora_alpha=16, lora_dropout=0.2,
            lora_last_n=4)
        xyz_m = np.random.RandomState(0).randn(24, 3).astype(np.float32) * 0.03
        model = build_corteg(24, 128, xyz_m, args)

        soft = model.backbone.channel_adapter.soft
        init = next(p for n, p in soft.named_parameters() if p.dim() >= 2)
        self.assertGreater(len(torch.unique(init.detach())), 1,
                           "the spatial adapter's init is a single constant — "
                           "knn_sigma=0 collapsed the KNN prior")

    def test_event_caches_are_keyed_by_seed(self):
        """seed and max_per_class choose WHICH events are drawn.

        The paper averages seeds 1, 2 and 42. If they are not in the cache key,
        all three reuse one event selection while reporting different seeds.
        """
        for runner in (BTB_RUNNER, FM_RUNNER):
            src = runner.read_text(encoding="utf-8")
            tag = src[src.index("    tag = "):]
            tag = tag[:tag.index(".npz")]
            self.assertIn("args.seed", tag, f"{runner.name}: cache key omits the seed")
            self.assertIn("max_per_class", tag,
                          f"{runner.name}: cache key omits max_per_class")

    def test_canonical_trials_are_not_all_trial000(self):
        """sub_1/2/6 are not trial000; defaulting there scores a different film."""
        import experiments.run_btb_classification as btb
        self.assertEqual(btb.CANONICAL_TRIAL["sub_1"], "trial001")
        self.assertEqual(btb.CANONICAL_TRIAL["sub_2"], "trial006")
        self.assertEqual(btb.CANONICAL_TRIAL["sub_6"], "trial004")
        self.assertEqual(len(btb.CANONICAL_TRIAL), 10)

    def test_electrode_selection_is_not_vendored(self):
        """PopT's selection has no redistribution licence; we must only link it."""
        self.assertEqual(list(REPO.glob("**/clean_laplacian.json")), [],
                         "PopT electrode selection must not be vendored")
        import experiments.run_btb_classification as btb
        os.environ.pop("POPT_REPO", None)
        with self.assertRaises(SystemExit) as cm:
            btb.clean_electrodes("sub_3")
        self.assertIn("PopulationTransformer", str(cm.exception))

    def test_splits_are_causal_and_embargoed(self):
        """Every fold: fit < val < test, with a gap wider than the embargo.

        The expected gap is the literal 7.0 s, NOT the imported EMBARGO_SEC:
        asserting against the constant means lowering the constant also lowers
        the assertion, and zeroing the embargo passes silently.
        """
        import numpy as np
        from data.braintreebank import forward_chaining_split, EMBARGO_SEC
        EXPECTED_EMBARGO = 7.0
        self.assertEqual(EMBARGO_SEC, EXPECTED_EMBARGO,
                         "embargo changed; the widest shared arm is Brant L=1 at 6.11 s")
        times = np.sort(np.random.RandomState(0).uniform(0, 7200, 1800))
        folds = forward_chaining_split(times, win_sec=1.5, n_folds=4, val_frac=0.15)
        self.assertGreaterEqual(len(folds), 1)
        for i, (fit, val, te) in enumerate(folds):
            self.assertLess(times[fit].max(), times[val].min(), f"fold {i} fit/val")
            self.assertLess(times[val].max(), times[te].min(), f"fold {i} val/test")
            self.assertGreater(times[val].min() - times[fit].max(), EXPECTED_EMBARGO)
            self.assertGreater(times[te].min() - times[val].max(), EXPECTED_EMBARGO)

    def test_leakage_is_detected(self):
        """The overlap assertion must fire on a deliberately leaking split."""
        import numpy as np
        from data.braintreebank import assert_no_window_overlap
        times = np.arange(0, 100, 0.5)          # 0.5 s apart, 1.5 s windows
        with self.assertRaises(AssertionError):
            assert_no_window_overlap(times, 1.5, np.arange(0, 50), np.arange(50, 200))

    def test_streams_yield_equal_token_counts(self):
        """128/16 == 200/25: the two streams must be fusable."""
        import numpy as np
        from data.braintreebank import corteg_features
        x = np.random.RandomState(0).randn(2, 4, int(1.5 * 2048)).astype(np.float32)
        lo, hi = corteg_features(x, fs=2048.0)
        self.assertEqual(lo.shape[-1] // 16, hi.shape[-1] // 25)


    def test_task_b_footprint_is_wider_than_task_a(self):
        """Task B centres a 5 s window, so its overlap footprint is 5.0, not 1.5.

        Passing Task A's 1.5 s to the split for a Task B run would under-embargo
        every fold boundary and leak.
        """
        from data.braintreebank import TILE_SEC, WIN_SEC, EMBARGO_SEC
        self.assertEqual(WIN_SEC, 5.0)
        self.assertEqual(TILE_SEC, 1.0)
        self.assertGreater(EMBARGO_SEC, WIN_SEC,
                           "embargo must exceed the widest shared footprint")

    def test_brant_fs_guard_fires_on_sub_9(self):
        """sub_9 records at ~1019 Hz; assuming 2048 doubles Brant's footprint."""
        from data.braintreebank import assert_brant_fs_fixed
        assert_brant_fs_fixed(2038.0)          # a normal subject passes
        with self.assertRaises(AssertionError):
            assert_brant_fs_fixed(1019.0)      # sub_9 must not pass silently

    def test_both_coordinate_frames_exist(self):
        """The FM arms select electrodes in voxel space, CORTEG in MNI.

        The two sets differ on several subjects, so mixing them silently changes
        which electrodes a published number was computed over.
        """
        import data.braintreebank as btb
        self.assertTrue(hasattr(btb, "load_localization"))      # voxel L/I/P
        self.assertTrue(hasattr(btb, "load_localization_mni"))  # shared MNI mm


class TestIeegFm(unittest.TestCase):
    """The intracranial-FM comparison arms."""

    def test_fm_script_flags_are_known_and_valid(self):
        spec = parser_spec(FM_RUNNER)
        self.assertTrue(FM_SCRIPTS, "no intracranial-FM script found")
        problems = []
        for sh in FM_SCRIPTS:
            for flag, val in script_flags(sh.read_text(encoding="utf-8")):
                if flag not in spec:
                    problems.append(f"{sh.name}: unknown flag {flag}")
                    continue
                info = spec[flag]
                if info["store_true"] and val is not None:
                    problems.append(f"{sh.name}: {flag} is store_true, got {val!r}")
                elif info["choices"] and val is not None and val not in info["choices"]:
                    problems.append(
                        f"{sh.name}: {flag}={val!r} not in {sorted(info['choices'])}")
        self.assertEqual(problems, [], "\n" + "\n".join(problems))

    def test_no_third_party_weights_or_code_are_vendored(self):
        """BrainBERT ships no LICENSE, so nothing of it may live in this repo."""
        for pat in ("**/*.pth", "**/stft_large*", "**/pre_model.py",
                    "**/clean_laplacian.json"):
            found = [f for f in REPO.glob(pat) if ".git" not in str(f)]
            self.assertEqual(found, [], f"third-party artefact vendored: {found}")

    def test_every_fm_is_reached_through_an_env_var(self):
        """No accessor may fall back to a path on the author's machine."""
        import ieeg_fm
        for getter in ("brainbert_repo", "brainbert_weights", "popt_repo",
                       "brant_src_dir", "brant_weights_dir"):
            for var in ("BRAINBERT_REPO", "BRAINBERT_WEIGHTS", "POPT_REPO",
                        "BRANT_SRC", "BRANT_WEIGHTS"):
                os.environ.pop(var, None)
            with self.assertRaises(SystemExit, msg=f"{getter} did not guard"):
                getattr(ieeg_fm, getter)()

    def test_task_b_window_is_wider_in_the_fm_runner(self):
        """Task B must use the 5 s footprint, or its folds under-embargo."""
        from experiments.run_ieeg_fm_baselines import ENDPOINTS
        self.assertEqual(ENDPOINTS["sentence_onset"]["win_sec"], 1.5)
        self.assertEqual(ENDPOINTS["word_nonword"]["win_sec"], 5.0)

    def test_brant_gets_its_native_context(self):
        """Brant's patch is 6 s; a 1.5 s window would be zero-padded silently.

        The 6 s window is also what sizes the shared embargo: its footprint is
        6.11 s once the resampler's filter edge is counted, and EMBARGO_SEC is 7.0.
        """
        from data.braintreebank import EMBARGO_SEC
        from experiments.run_ieeg_fm_baselines import BRANT_WINDOW, window_for
        import ieeg_fm

        self.assertEqual(BRANT_WINDOW["win_sec"], 6.0)
        self.assertEqual(BRANT_WINDOW["pre_sec"], -4.5)
        # one full patch, exactly
        self.assertEqual(ieeg_fm.BRANT_PATCH_LEN / ieeg_fm.BRANT_FS,
                         BRANT_WINDOW["win_sec"])
        # Brant ignores the endpoint window; the other arms do not
        for ep in ("sentence_onset", "word_nonword"):
            self.assertEqual(window_for("brant", ep), BRANT_WINDOW)
            self.assertNotEqual(window_for("brainbert", ep), BRANT_WINDOW)
        self.assertGreater(EMBARGO_SEC, BRANT_WINDOW["win_sec"] + 0.11,
                           "embargo must exceed Brant's footprint incl. filter edge")

    def test_oracle_arm_ships_its_caveat(self):
        """single_elec_max is an oracle; every place that says so must keep saying it.

        Checked per-location rather than "is 0.53 anywhere in the file": the
        string appears in both the module docstring and the emitted caveat, so a
        file-wide search stays true when one of them is corrupted.
        """
        src = FM_RUNNER.read_text(encoding="utf-8")
        head = src[:src.index('"""', 3)]                    # module docstring only
        self.assertIn("0.53", head, "the docstring must state the inflated null")

        caveat = src[src.index('out["caveat"]'):]
        caveat = caveat[:caveat.index(")\n")]
        self.assertIn("0.53", caveat, "the emitted caveat must state the null")
        self.assertIn("single_elec_mean", caveat,
                      "the caveat must point at the non-oracle comparison")

        readme = (REPO / "README.md").read_text(encoding="utf-8")
        self.assertIn("oracle", readme.lower())
        self.assertIn("0.53", readme, "README must state the inflated null")

    def test_adapters_have_a_caller(self):
        """ieeg_fm.py must not be dead code."""
        callers = [f for f in REPO.glob("experiments/*.py")
                   if "ieeg_fm" in f.read_text(encoding="utf-8")]
        self.assertTrue(callers, "ieeg_fm.py has no caller")


class TestConfigs(unittest.TestCase):
    def test_configs_declare_a_pretrained_path(self):
        for cfg in sorted((REPO / "configs").glob("*.json")):
            with self.subTest(cfg=cfg.name):
                d = json.loads(cfg.read_text(encoding="utf-8"))
                self.assertIn("pretrained", d)
                self.assertTrue(str(d["pretrained"].get("path", "")).strip())


class TestPaths(unittest.TestCase):
    """CORTEG_* names are what the docs tell users to set; both must work."""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       ("CORTEG_DATA_ROOT", "ECOG_DATA_ROOT",
                        "CORTEG_OUTPUT_ROOT", "ECOG_OUTPUT_ROOT")}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_corteg_and_ecog_names_both_work(self):
        import importlib
        import paths as paths_mod
        importlib.reload(paths_mod)
        os.environ["ECOG_DATA_ROOT"] = "/tmp/ecog"
        self.assertEqual(paths_mod.get_data_root(), "/tmp/ecog")
        os.environ["CORTEG_DATA_ROOT"] = "/tmp/corteg"
        self.assertEqual(paths_mod.get_data_root(), "/tmp/corteg",
                         "CORTEG_DATA_ROOT must take precedence")

    def test_explicit_override_wins(self):
        import importlib
        import paths as paths_mod
        importlib.reload(paths_mod)
        os.environ["CORTEG_DATA_ROOT"] = "/tmp/corteg"
        self.assertEqual(paths_mod.get_data_root("/explicit"), "/explicit")


class TestDocs(unittest.TestCase):
    def test_docs_do_not_reference_removed_scripts(self):
        stale = []
        for doc in ("README.md",):
            text = (REPO / doc).read_text(encoding="utf-8")
            for ref in re.findall(r'scripts/[A-Za-z0-9_]+\.sh', text):
                if not (REPO / ref).exists():
                    stale.append(f"{doc}: {ref}")
        self.assertEqual(stale, [], "\n" + "\n".join(stale))

    def test_readme_does_not_advertise_the_broken_eval_recipe(self):
        text = (REPO / "README.md").read_text(encoding="utf-8")
        # Only fenced code blocks are recipes a reader would copy; surrounding
        # prose may legitimately name the broken pairing in order to warn about it.
        for block in re.findall(r"```[a-z]*\n(.*?)```", text, re.S):
            if "--finetune_from" in block:
                self.assertNotIn(
                    "--train_mode per_subject", block,
                    "README.md ships a recipe pairing --finetune_from with "
                    "--train_mode per_subject, which never reads the checkpoint",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
