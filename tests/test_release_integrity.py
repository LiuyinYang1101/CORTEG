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
SCRIPTS = sorted((REPO / "scripts").glob("*.sh"))
RUNNER_CALL = "python -m experiments.run_regression_hilo_clean"

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
            yield tok, (None if (nxt is None or nxt.startswith("--")) else nxt)


def _arrays(text: str):
    """Every `NAME=( ... )` bash array in `text`, as {name: body}."""
    return {m.group(1): m.group(2)
            for m in re.finditer(r"^(\w+)=\((.*?)\)", text, re.M | re.S)}


def _functions(text: str):
    """Every `name() { ... }` shell function in `text`, as {name: body}."""
    return {m.group(1): m.group(2)
            for m in re.finditer(r"^(\w+)\(\)\s*\{\n(.*?)^\}", text, re.M | re.S)}


def _invocations(text: str):
    """Each runner invocation in `text`, with bash expansions resolved.

    Handles the direct `python -m experiments...` form and any wrapper function
    (`run`, `run_no_fuser`) that calls it, inlining the function body and every
    `"${ARRAY[@]}"` it expands. Resolving the expansion rather than grepping the
    file is what makes the per-call check sound: dropping a flag from a shared
    array then shows up on every call that used it.
    """
    arrays, funcs = _arrays(text), _functions(text)
    runners = {n for n, b in funcs.items() if RUNNER_CALL in b}

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
        starts = RUNNER_CALL in line or called is not None
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

        gt = Path(paths_mod.get_output_root()) / "stanford_best_lora_adapter" / "results_pooled.json"
        if not gt.exists():
            self.skipTest("no reference args available")
        base = json.loads(gt.read_text(encoding="utf-8"))["args"]
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
        gt = Path(__import__("paths").get_output_root()) / "stanford_best_lora_adapter" / "results_pooled.json"
        if not gt.exists():
            self.skipTest("no reference args available")
        base = json.loads(gt.read_text(encoding="utf-8"))["args"]
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
