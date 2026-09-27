"""Load the released CORTEG adapter into a ready-to-run model.

The released checkpoint holds only the ~297 K trainable parameters (LoRA,
spatial adapter, LayerNorm, head). The frozen ST-EEGFormer backbone is a
separate 376 MB download -- see the Checkpoints section of README.md.

Why this module exists rather than `--finetune_from`: that path loads with
`strict=False`, which is right for cross-task fine-tuning but wrong for a
release. If a user rebuilds a slightly different architecture, `strict=False`
loads almost nothing, reports success, and produces a randomly-initialised
model that looks trained. `load_corteg` refuses instead: every tensor in the
checkpoint must find a home, or it raises.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(REPO, "checkpoints", "corteg_stanford_pooled.pt")
DEFAULT_MANIFEST = os.path.join(REPO, "checkpoints", "corteg_stanford_pooled.json")


def verify_checkpoint(checkpoint: str = DEFAULT_CKPT,
                      manifest: str = DEFAULT_MANIFEST) -> dict:
    """Check the checkpoint against its manifest (size + sha256). Returns the manifest."""
    import hashlib

    with open(manifest, encoding="utf-8") as fh:
        man = json.load(fh)
    with open(checkpoint, "rb") as fh:
        blob = fh.read()
    if len(blob) != man["bytes"]:
        raise ValueError(f"{checkpoint}: {len(blob)} bytes, manifest says {man['bytes']}")
    got = hashlib.sha256(blob).hexdigest()
    if got != man["sha256"]:
        raise ValueError(f"{checkpoint}: sha256 {got[:16]}… != manifest {man['sha256'][:16]}…")
    return man


def load_corteg(
    C_in: int,
    T_in: int,
    ecog_xyz_mm: np.ndarray,
    d_out: int = 5,
    device: str = "cpu",
    checkpoint: str = DEFAULT_CKPT,
    manifest: str = DEFAULT_MANIFEST,
    backbone_config: Optional[str] = None,
    verify: bool = True,
):
    """Rebuild the released CORTEG model and load its weights.

    Args:
        C_in: number of electrodes.
        T_in: window length in samples.
        ecog_xyz_mm: (C_in, 3) electrode coordinates in MILLIMETRES (build_model
            converts to metres internally); the KNNSoftFourier
            adapter is conditioned on these, so they are required, not optional.
        d_out: readout dimension (5 for Stanford fingerflex).
        backbone_config: path to the ST-EEGFormer config; defaults to the one
            recorded in the manifest.

    Raises:
        RuntimeError: if any checkpoint tensor is unused or any shape disagrees.
    """
    from experiments.run_regression_hilo_clean import build_model
    from models.steegformer.probe import configure_lora_lastn_probe

    man = verify_checkpoint(checkpoint, manifest) if verify else json.load(
        open(manifest, encoding="utf-8"))

    args = SimpleNamespace(**man["build_args"])
    args.model_kwargs_json = backbone_config or os.path.join(
        REPO, man["build_args"]["model_kwargs_json"])
    args.no_pretrained = False

    xyz = np.asarray(ecog_xyz_mm, dtype=np.float32)
    if xyz.shape != (C_in, 3):
        raise ValueError(f"ecog_xyz_mm must be ({C_in}, 3), got {xyz.shape}")

    model = build_model(args, C_in=C_in, T_in=T_in, ecog_xyz_m=xyz, d_out=d_out)

    # LoRA is injected by the training setup, not by build_model, so a freshly
    # built model has no .A/.B tensors for the checkpoint to land in. Mirror the
    # runner's call exactly -- the ranks and targets come from the manifest.
    configure_lora_lastn_probe(
        model,
        n_last=int(args.lora_last_n),
        r=int(args.lora_r),
        alpha=int(args.lora_alpha),
        dropout=float(args.lora_dropout),
        targets=tuple(t.strip() for t in args.lora_targets.split(",") if t.strip()),
    )

    sd = torch.load(checkpoint, map_location="cpu")

    # The readout head is built lazily on the first forward pass, so it does not
    # exist yet; infer its input dim from the checkpoint and materialise it.
    if getattr(model.head, "head", None) is None:
        key = next((k for k in sd if k.startswith("head.head.") and k.endswith(".weight")), None)
        if key is not None:
            model.head.head = model.head._build_head(sd[key].shape[-1], torch.device("cpu"))

    model_sd = model.state_dict()
    missing = [k for k in sd if k not in model_sd]
    wrong = [(k, tuple(sd[k].shape), tuple(model_sd[k].shape))
             for k in sd if k in model_sd and sd[k].shape != model_sd[k].shape]
    if missing or wrong:
        raise RuntimeError(
            "Released checkpoint does not fit the rebuilt model — the architecture "
            "drifted from the weights.\n"
            f"  unknown keys: {missing[:6]}{' …' if len(missing) > 6 else ''}\n"
            f"  shape clashes: {wrong[:4]}{' …' if len(wrong) > 4 else ''}\n"
            "  Check that backbone_config matches architecture_args in the manifest."
        )

    # strict=False is safe *now*: we just proved every checkpoint tensor matches.
    # The frozen backbone legitimately has no entry here, which is the whole point.
    model.load_state_dict(sd, strict=False)
    n = sum(v.numel() for v in sd.values())
    if n != man["trainable_params"]:
        raise RuntimeError(f"loaded {n} params, manifest says {man['trainable_params']}")

    # The spatial adapter needs the electrode coordinates on every forward pass;
    # stash them so `predict` can supply them and callers need not know.
    model = model.to(device).eval()
    # The model wants (B, C, 3) in METRES at call time, while build_model above
    # wants (C, 3) in millimetres for the adapter bank. Store the metre copy.
    model._corteg_xyz_m = torch.from_numpy(xyz / 1000.0).float().to(device)
    model._corteg_stream = man["build_args"].get("stream", "both")
    return model


@torch.no_grad()
def predict(model, x_lo, x_hi=None, batch_size: int = 32) -> np.ndarray:
    """Run a loaded CORTEG model over windows.

    CORTEG is dual-stream. Both streams cover the same 1 s window but are
    sampled differently, and the patch sizes are chosen so they yield the same
    token count (128 / 16 == 200 / 25 == 8 patches per electrode):

        x_lo: (N, C, 128)  broadband signal at 128 Hz
        x_hi: (N, C, 200)  high-gamma envelope feature

    Args:
        x_lo: (N, C, 128) low-frequency stream.
        x_hi: (N, C, 200) high-gamma stream. Required unless the model was
            built with stream="lo_only".
        batch_size: windows per forward pass.

    Returns:
        (N, d_out) predictions.
    """
    xyz = getattr(model, "_corteg_xyz_m", None)
    if xyz is None:
        raise RuntimeError("model was not produced by load_corteg()")
    stream = getattr(model, "_corteg_stream", "both")
    if x_hi is None and stream != "lo_only":
        raise ValueError(
            f"this model was built with stream={stream!r} and needs x_hi "
            "(the high-gamma stream); see data/stanford_preprocessing/"
        )
    device = xyz.device
    lo = torch.as_tensor(np.asarray(x_lo, dtype=np.float32))
    hi = None if x_hi is None else torch.as_tensor(np.asarray(x_hi, dtype=np.float32))
    if hi is not None and len(hi) != len(lo):
        raise ValueError(f"x_lo has {len(lo)} windows but x_hi has {len(hi)}")
    out = []
    for i in range(0, len(lo), batch_size):
        chunk = lo[i:i + batch_size].to(device)
        y = model(
            chunk,
            x_hi=None if hi is None else hi[i:i + batch_size].to(device),
            # one copy per window: the adapter is conditioned per sample
            ecog_xyz=xyz.unsqueeze(0).expand(chunk.shape[0], -1, -1),
        )
        out.append((y[0] if isinstance(y, (tuple, list)) else y).float().cpu().numpy())
    return np.concatenate(out, axis=0)
