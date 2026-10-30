"""Export the fixed epoch 29 of the Update 3 run in the Update 1 format (protocol: "Exportálás").

- State dict: the training checkpoint's model without the two ordinal parameters (_ord_gap, _ord_b0), so the published
  Update 1 loader builds it unchanged.
- Meta: as tools/export_weights.py with --run (thresholds r90 / r95 / r98 from the validation frames of epoch 29), plus
  "ordinal": the grid T and the learned thresholds b (px reading, common.measured_L).
- Parity: the training model and the exported file give the same image logit and cell logits on the probe frame
  (GPU, bfloat16 autocast), and the meta's px reading equals the training model's measure().

usage: python experiments/u3r3_eval/export.py
Output: weights/update3_small_p99_ep029.pt, experiments/u3r3_eval/data/export.json
"""
from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image

import common as C
from blurtrainer.ood import frames_thresholds
from blurtrainer.train import _model


def main():
    C.DATA.mkdir(parents=True, exist_ok=True)
    src = C.RUN / "epochs" / f"epoch_{C.EPOCH:03d}.pt"
    ck = torch.load(src, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    assert cfg["epochs"] == 30 and cfg.get("ordinal_T") and cfg.get("ordinal_anchor", 1.0) is None, cfg
    ref = _model(cfg, None, False)
    ref.load_state_dict(ck["model"])
    ref = ref.cuda().eval()
    b = [float(v) for v in ref.ord_b().detach().cpu()]
    T = [float(t) for t in cfg["ordinal_T"]]
    th = frames_thresholds(C.RUN, C.EPOCH)
    meta = {"input": cfg["input_norm"], "input_description": "edge fingerprint (m09 residual, p99)",
            "backbone": f"{cfg['backbone']} ({'ImageNet-pretrained' if cfg.get('weights') else 'from scratch'}), stride 16",
            "epoch": C.EPOCH, "epochs_total": cfg["epochs"],
            "thresholds": {k: float(th[k]) for k in ("r90", "r95", "r98")},
            "thresholds_meaning": "p_blurred >= threshold -> blurred; catches 90 / 95 / 98% of the blurred frames of the validation split",
            "val_frames_auc": float(th["frames_auc"]),
            "val_sharp_kept": {k: float(v) for k, v in th["frames_sharp_kept"].items() if k.startswith("sharp_kept")},
            "license": "MIT (see LICENSE); trained on non-public material",
            "arch": cfg["backbone"], "fingerprint": "fp32" if cfg.get("fp32_input") else "bf16-rounded", "run": C.RUN.name,
            "ordinal": {"T": T, "b": b, "anchor": "floating (no threshold pinned to 0)",
                        "measure": "L = sum_k sigmoid(logit - b_k) * (t_k - t_{k-1}), t_0 = 0; image or cell logit"}}
    assert not cfg.get("mask_leak_fix"), "Update 3 trains without masks"
    sd = {k: v.contiguous() for k, v in ck["model"].items() if not k.startswith("_ord_")}
    assert len(sd) == len(ck["model"]) - 2
    C.U3_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sd, "meta": meta}, C.U3_FILE)
    new, meta2 = C.U1.load_detector(C.U3_FILE, "cuda")
    probe = torch.from_numpy(np.asarray(Image.open(C.PROBE).convert("RGB")).copy()).permute(2, 0, 1)[None].cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        o_ref, o_new = ref(probe), new(probe)
        L_ref = float(ref.measure(o_ref["image"]).float())
    x, y = float(o_ref["image"].float()), float(o_new["image"].float())
    dl = float((o_ref["local"].float() - o_new["local"].float()).abs().max())
    L_meta = float(C.measured_L(y, meta2))
    rec = {"source": str(src.relative_to(C.ROOT)), "file": str(C.U3_FILE.relative_to(C.ROOT)), "sha256": C.sha256(C.U3_FILE),
           "size_bytes": C.U3_FILE.stat().st_size, "meta": meta,
           "parity_probe": {"image_logit_train": x, "image_logit_export": y, "max_abs_cell_logit_difference": dl,
                            "L_train_measure": L_ref, "L_from_meta": L_meta}}
    assert abs(x - y) < 1e-3 and dl < 1e-2 and abs(L_ref - L_meta) < 1e-3, rec["parity_probe"]
    (C.DATA / "export.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    C.log(f"{C.U3_FILE.name}: {rec['size_bytes'] / 2**20:.0f} MiB, sha256 {rec['sha256'][:12]}, thresholds {meta['thresholds']}, "
          f"val AUC {meta['val_frames_auc']:.4f}")
    C.log(f"  parity: logit {x:.5f} / {y:.5f}, cell max diff {dl:.2e}, L {L_ref:.4f} / {L_meta:.4f} px")
    C.log(f"  b (px grid {T}): " + " ".join(f"{v:.2f}" for v in b))


if __name__ == "__main__":
    main()
