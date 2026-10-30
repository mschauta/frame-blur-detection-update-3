"""Shared by the Update 3 evaluation (docs/kiserlet_u3r3_gop_letra.md, "Kiértékelés"): paths, the two weight files,
the published Update 1 loader, and the px reading of the ordinal head.

The Update 3 file is in the Update 1 format: the state dict without the two ordinal parameters, loaded with the
published Update 1 loader (frame-blur-detection-update-1 blurdetect), so its image logit is the frame decision exactly
as Update 1 and the FrameBlurStudio read it. The ordinal thresholds b_k and the grid t_k are in meta["ordinal"]; the
measured length of a logit s is L = sum_k sigmoid(s - b_k) (t_k - t_{k-1}), t_0 = 0 (blurtrainer.model.measured_L).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
UP = ROOT.parent
DATA = HERE / "data"
RUN = ROOT / "runs" / "update3_small_gop_ladder_v1"
SMALL_RUN = ROOT / "runs" / "round3_convnext_small_imagenet_p99_v8_v1"
EPOCH = 29
U3_FILE = ROOT / "weights" / "update3_small_p99_ep029.pt"
SMALL_FILE = UP / "frame-blur-detection-update-1" / "weights" / "round3_small_p99_ep029.pt"
SMALL_SHA = "b715a163f9c6927464d97b7002967c8361c6a0de2b87e3103dc72a7ba733d48b"
PROBE = Path("F:/RGB Mesh/demo_processed/frames_1080_h264/07_2b_00170.png")
SEED = 20261030

_spec = importlib.util.spec_from_file_location("blurdetect_u1", UP / "frame-blur-detection-update-1" / "blurdetect" / "model.py")
U1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(U1)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_head(path: Path, n: int = 64 * 2**20) -> str:
    """Digest of the first 64 MiB (identifies a video file without reading all of it)."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read(n)).hexdigest()


def load_u3(device="cuda"):
    """(model, meta) of the exported Update 3 file; its digest is recorded at export (data/export.json)."""
    rec = json.loads((DATA / "export.json").read_text(encoding="utf-8"))
    assert sha256(U3_FILE) == rec["sha256"], "the Update 3 file changed since the export"
    model, meta = U1.load_detector(U3_FILE, device)
    assert meta["epoch"] == EPOCH and "ordinal" in meta, meta
    return model, meta


def load_small(device="cuda"):
    assert sha256(SMALL_FILE) == SMALL_SHA
    model, meta = U1.load_detector(SMALL_FILE, device)
    assert meta["epoch"] == EPOCH and meta["input"] == "p99", meta
    return model, meta


def measured_L(logit, meta) -> np.ndarray:
    """px of an Update 3 logit (any shape), from the exported ordinal thresholds."""
    T = np.asarray(meta["ordinal"]["T"], float)
    b = np.asarray(meta["ordinal"]["b"], float)
    dT = np.diff(T, prepend=0.0)
    s = np.asarray(logit, float)[..., None]
    return (dT / (1 + np.exp(-(s - b)))).sum(-1)


def forward(model, x):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model(x)


def logit_threshold(meta, point="r95") -> float:
    t = meta["thresholds"][point]
    return float(np.log(t / (1 - t)))
