"""Protocol item 4 (cell maps): the cell-level motion witness of R4T-1 9.6, and the cell px over training.

witness: the same 2,000 validation frames as R4T-1 9.6 (round3 index, val split, ordered by h("cellcheck")); per
    160 px measurement cell with >= 500 textured pixels: subject motion (camera-compensated travel_px), total motion
    (uncompensated) and static detail (detail_p95); the model's 16 px map averaged over the same block. Main measure:
    the within-frame Spearman (do the two maps rank the cells of one frame the same way). Update 3 ep029 and the
    round3 Small ep029, both from their training checkpoints (training code; no mask).
checkpoints: Update 3 at epochs 0 / 12 / 15 / 29 on (1) the same 2,000 frames and (2) every steady camera pan of the
    validation frames (napló 2026-10-30 04:40); per frame the image px and the inner cells' px (median, p90, max),
    each epoch with its own ordinal thresholds. Epoch-to-epoch noise applies (epoch 15 was a low point).

usage: python experiments/u3r3_eval/cells.py witness
       python experiments/u3r3_eval/cells.py checkpoints
Output: data/cells_witness.json, data/cells_checkpoints.csv, data/cells_checkpoints.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import common as C
import probe_round3 as P
from blurtrainer.metrics import spearman

DB = "file:<master>/master.db?mode=ro"
SCRATCH = Path("<scratch>")          # the steady-pan list of napló 2026-10-30 04:40 (u3_pan_blur.npz, u3_val_motion.csv; not published)
EPOCHS = (0, 12, 15, 29)


def val_rows(n=2000):
    rows = [r for r in csv.DictReader(open(C.ROOT / "data" / "round3_index.csv", encoding="utf-8"))
            if r["source"] == "frames" and r["split"] == "val"]
    return sorted(rows, key=lambda r: P.h("cellcheck", r["sample_id"]))[:n]


def maps(model, rgb):
    x = torch.from_numpy(np.ascontiguousarray(rgb)[None]).permute(0, 3, 1, 2).contiguous().cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x)
    return out


def witness(a):
    rows = val_rows()
    near = {x["sample_id"]: x["closeup"] == "1"
            for x in csv.DictReader(open(C.ROOT / "data" / "closeup_frames_round3_index.csv", encoding="utf-8"))}
    models = {"u3": P.load_model_from(C.RUN, C.EPOCH), "small": P.load_model_from(C.SMALL_RUN, C.EPOCH)}
    con = sqlite3.connect(DB, uri=True)
    res = {k: {"comp": [], "unc": [], "det": [], "pm": [], "pc": [], "pu": []} for k in models}
    near_flag, n_cells = [], 0
    for n, r in enumerate(rows):
        fid = r["sample_id"].split("/", 1)[1]
        ev = con.execute("SELECT payload FROM frame_measurement_evidence WHERE frame_id = ?", (fid,)).fetchone()
        if not ev:
            continue
        mc = json.loads(zlib.decompress(ev[0]))["motion_cells"]
        if mc["compensated"].get("status") != "measured":
            continue
        unc = {tuple(c["bbox"]): c for c in mc["uncompensated"]["cells"]}
        cells = []
        for c in mc["compensated"]["cells"]:
            if c["status"] != "measured" or c.get("structured_count", 0) < 500:
                continue
            u = unc.get(tuple(c["bbox"]))
            if not u or u.get("travel_px") is None or c.get("travel_px") is None:
                continue
            x0, y0, x1, y1 = c["bbox"]
            cells.append((x0 // 16, y0 // 16, -(-x1 // 16), -(-y1 // 16), c["travel_px"], u["travel_px"], c.get("detail_p95") or np.nan))
        if len(cells) < 8:
            continue
        rgb = np.asarray(Image.open(r["png"]).convert("RGB"))
        tc, tu, td = (np.array([c[i] for c in cells], float) for i in (4, 5, 6))
        near_flag.append(near.get(r["sample_id"], False))
        n_cells += len(cells)
        for name, m in models.items():
            out = maps(m, rgb)
            loc, val = out["local"][0, 0].float().cpu().numpy(), out["cell_valid"][0, 0].cpu().numpy()
            v = np.array([np.mean(loc[a1:b1, a0:b0][val[a1:b1, a0:b0]]) if val[a1:b1, a0:b0].any() else np.nan
                          for a0, a1, b0, b1, *_ in cells])
            ok = np.isfinite(v)
            res[name]["comp"].append(spearman(v[ok], tc[ok]) if ok.sum() >= 8 else np.nan)
            res[name]["unc"].append(spearman(v[ok], tu[ok]) if ok.sum() >= 8 else np.nan)
            okd = ok & np.isfinite(td)
            res[name]["det"].append(spearman(v[okd], td[okd]) if okd.sum() >= 8 else np.nan)
            res[name]["pm"].append(v[ok]); res[name]["pc"].append(tc[ok]); res[name]["pu"].append(tu[ok])
        if (n + 1) % 250 == 0:
            C.log(f"  {n + 1} / {len(rows)}")
    nf = np.array(near_flag)
    out = {"frames": int(len(nf)), "near_frames": int(nf.sum()), "cells": n_cells,
           "what": "R4T-1 9.6 repeated: within-frame Spearman between the model's cell map (16 px logits averaged over the "
                   "160 px block) and the measured motion; textured (>= 500 px) measured cells", "models": {}}
    rng = np.random.default_rng(C.SEED)
    for name, d in res.items():
        c, u, dd = (np.array(d[k]) for k in ("comp", "unc", "det"))
        pm, pc, pu = (np.concatenate(d[k]) for k in ("pm", "pc", "pu"))
        boot = [np.nanmedian(c[rng.integers(len(c), size=len(c))]) for _ in range(2000)]
        out["models"][name] = {"subject_within_frame_median": float(np.nanmedian(c)),
                               "subject_within_frame_median_ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
                               "subject_not_near": float(np.nanmedian(c[~nf])), "subject_near": float(np.nanmedian(c[nf])),
                               "total_within_frame_median": float(np.nanmedian(u)),
                               "detail_within_frame_median": float(np.nanmedian(dd)),
                               "pooled_subject": float(spearman(pm, pc)), "pooled_total": float(spearman(pm, pu))}
    du = np.array(res["u3"]["comp"]) - np.array(res["small"]["comp"])
    boot = [np.nanmedian(du[rng.integers(len(du), size=len(du))]) for _ in range(2000)]
    out["u3_minus_small_subject_within_frame"] = {"median_of_differences": float(np.nanmedian(du)),
                                                  "ci95": np.percentile(boot, [2.5, 97.5]).tolist()}
    out["round3_base_r4t1_reference"] = {"subject_within_frame_median": 0.339, "total": 0.332, "detail": -0.179,
                                         "pooled_subject": 0.507}
    (C.DATA / "cells_witness.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for name, e in out["models"].items():
        C.log(f"{name}: frame-en belül alany {e['subject_within_frame_median']:.3f} {np.round(e['subject_within_frame_median_ci95'], 3).tolist()} "
              f"(nem közeli {e['subject_not_near']:.3f}, közeli {e['subject_near']:.3f}), teljes {e['total_within_frame_median']:.3f}, "
              f"részletesség {e['detail_within_frame_median']:.3f}; összesítve {e['pooled_subject']:.3f} / {e['pooled_total']:.3f}")
    C.log(f"U3 − Small (alany, frame-en belül): {out['u3_minus_small_subject_within_frame']}")


def checkpoints(a):
    sample = val_rows()
    ev = {r["sample_id"]: r for r in csv.DictReader(open(C.RUN / "eval" / f"val_epoch_{C.EPOCH:03d}.csv", encoding="utf-8"))
          if r["source"] == "frames"}
    Z = np.load(SCRATCH / "u3_pan_blur.npz")
    mot = {r["sample_id"]: r for r in csv.DictReader(open(SCRATCH / "u3_val_motion.csv", encoding="utf-8"))}
    pans = [str(i) for i, p in zip(Z["ids"], Z["is_pan"]) if p]
    items = [(r["sample_id"], "sample") for r in sample] + [(i, "pan") for i in pans]
    C.log(f"checkpointok {EPOCHS}: {len(sample)} minta-frame + {len(pans)} egyenletes pásztázás")
    rows = []
    for ep in EPOCHS:
        m = P.load_model_from(C.RUN, ep)
        T = m.ord_T
        b = m.ord_b().detach()
        dT = torch.diff(T, prepend=T.new_zeros(1))
        for n, (sid, kind) in enumerate(items):
            rgb = np.asarray(Image.open(ev[sid]["png"]).convert("RGB"))
            out = maps(m, rgb)
            inner = out["local"].float()[0, 0, 1:-1, 1:-1]
            Lc = (torch.sigmoid(inner.unsqueeze(-1) - b) * dT).sum(-1).flatten()
            q = torch.quantile(Lc, torch.tensor([0.5, 0.9], device="cuda"))
            Li = float(m.measure(out["image"].float()))
            rows.append({"epoch": ep, "sample_id": sid, "kind": kind, "code": ev[sid]["code"],
                         "cam": mot.get(sid, {}).get("cam", ""), "L": Li, "Lc_med": float(q[0]), "Lc_p90": float(q[1]),
                         "Lc_max": float(Lc.max()), "logit": float(out["image"].float())})
        C.log(f"  epoch {ep} kész")
        del m
        torch.cuda.empty_cache()
    with open(C.DATA / "cells_checkpoints.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = {}
    for ep in EPOCHS:
        e = {}
        for code in ("sss", "ssb", "bsb", "bbb"):
            g = [r for r in rows if r["epoch"] == ep and r["kind"] == "sample" and r["code"] == code]
            if len(g) >= 20:
                e[f"sample/{code}"] = {k: float(np.median([r[k] for r in g])) for k in ("L", "Lc_med", "Lc_p90", "Lc_max")} | {"n": len(g)}
        for lo, hi in ((4, 8), (8, 16), (16, 24), (24, 999)):
            g = [r for r in rows if r["epoch"] == ep and r["kind"] == "pan" and r["cam"] and lo <= float(r["cam"]) < hi]
            if len(g) >= 20:
                e[f"pan/{lo}-{hi}"] = {k: float(np.median([r[k] for r in g])) for k in ("L", "Lc_med", "Lc_p90", "Lc_max")} | {"n": len(g)}
        res[f"epoch_{ep}"] = e
    (C.DATA / "cells_checkpoints.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for ep, e in res.items():
        C.log(f"{ep}: " + "; ".join(f"{k} L {v['L']:.2f} cella p90 {v['Lc_p90']:.2f} max {v['Lc_max']:.2f}" for k, v in e.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("witness", "checkpoints"))
    a = ap.parse_args()
    {"witness": witness, "checkpoints": checkpoints}[a.cmd](a)


if __name__ == "__main__":
    main()
