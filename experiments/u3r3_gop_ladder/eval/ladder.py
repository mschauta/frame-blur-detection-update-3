"""Protocol item 3: the photo ladder.

(a) The published Update 1 ladder (R4S-1): the same 300 unseen etalon photos, one direction per photo, L = 0 ... 30 px
    (17 lengths), single-frame H.264 CRF 23 (tools/probe_round3._ladder_job). Update 3 against the Small's R4S-1
    scores of the same images: the 50% flip, the 0-3 px pair order, strict increase over 0-1-2-3 px; for Update 3 also
    the measured px.
(b) The same ladder in GOP clips: the first 100 of these photos, the same direction and lengths, 8-frame clips through
    the training's coding chain (synth._gop_encoder) in x264 / x265 / AV1, I-only and IPB, at the PSNR-matched middle
    CRF of each list (x264 23, x265 26, AV1 38; no film grain); static, and a pan of 2 px per frame along the blur (the
    training's pans are 0-4 px). Every frame is scored (Update 3 and the Small); the frame types are read from the
    stream. Measures: the measured px against the true L (MAE, Spearman) by codec and frame type.

usage: python experiments/u3r3_eval/ladder.py a
       python experiments/u3r3_eval/ladder.py b [--workers 14]     (resumable)
       python experiments/u3r3_eval/ladder.py report
Output: data/ladder_a.npz, data/ladder_b.csv, data/ladder.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from scipy.stats import spearmanr

import common as C

ETALON_ALBUMS = ()          # the two etalon albums' bin names (withheld)
import eval_v9 as E
import probe_round3 as P
import r4s1_small_base as S

A_OUT, B_OUT = C.DATA / "ladder_a.npz", C.DATA / "ladder_b.csv"
B_PHOTOS, B_PAN = 100, 2
B_CRF = {"x264": 23, "x265": 26, "av1": 38}
B_STRUCT = ("I", "IPB")
B_COLS = ["photo", "L", "angle", "codec", "struct", "motion", "k", "type", "psnr", "u3_logit", "u3_L", "small_logit"]


def ladder_rows():
    from blurtrainer.data import read_index
    rows = [r for r in read_index(C.ROOT / "data" / "round3_index.csv") if r["source"] == "photo" and r["synth"] == "sharp"
            and r["split"] in ("val", "test") and r["shot_bin"] in ETALON_ALBUMS]
    rows.sort(key=lambda r: P.h("ladder", r["sample_id"]))
    rows = rows[:300]
    return [(r, P.ANGLES[P.h("angle", r["sample_id"]) % len(P.ANGLES)]) for r in rows]


def run_a(a):
    jobs = ladder_rows()
    small = np.load(S.OUT / "r4s1_ladder.npz")
    assert [j[0]["sample_id"] for j in jobs] == list(small["sample_id"])
    model, meta = C.load_u3()
    res = np.zeros((len(jobs), len(P.LADDER_L)))
    C.log(f"(a) Update 1-létra: {len(jobs)} fotó × {len(P.LADDER_L)} hossz, H.264 CRF 23, Update 3")
    with ProcessPoolExecutor(10) as ex:
        for i, images in enumerate(P.bounded_map(ex, P._ladder_job, jobs)):
            for j0 in range(0, len(images), 8):
                x = torch.from_numpy(np.stack(images[j0:j0 + 8])).permute(0, 3, 1, 2).contiguous().cuda()
                res[i, j0:j0 + 8] = C.forward(model, x)["image"].float().cpu().numpy()
            if (i + 1) % 50 == 0:
                C.log(f"  {i + 1} / {len(jobs)}")
    np.savez_compressed(A_OUT, L=np.array(P.LADDER_L), sample_id=small["sample_id"], u3=res)
    C.log(f"kész: {A_OUT}")


def _clip_job(job):
    """One 8-frame clip of a ladder photo: blur, (pan), code, decode; frame types from ffprobe."""
    from blurtrainer import synth
    sid, row, angle, L, codec, struct, motion = job
    rgb = P.load_photo(row)
    blurred = synth.camera_blur(rgb, L, angle)
    v = B_PAN if motion == "pan" else 0
    t = math.radians(angle)
    rec = {"dx": int(round(v * math.cos(t))), "dy": int(round(-v * math.sin(t)))}
    wins = [synth.gop_window(rgb.shape[:2], rec, j) for j in range(synth.GOP_N)]
    clip = np.stack([blurred[y:y + h, x:x + w] for y, x, h, w in wins])
    n, hh, ww = clip.shape[:3]
    data, pix = np.ascontiguousarray(clip).tobytes(), "rgb24"
    if data[:3] == b"ID3":
        data, pix = np.ascontiguousarray(clip[..., ::-1]).tobytes(), "bgr24"
    enc_args, fmt, box = synth._gop_encoder(codec, struct, B_CRF[codec], 0)
    enc = synth._run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{ww}x{hh}", "-r", "30", "-i", "-",
                      "-vf", f"scale=out_color_matrix=bt709:out_range=tv,format={fmt}", *enc_args, "-f", box, "-"], data)
    pr = subprocess.run(["ffprobe", "-v", "error", "-f", box, "-i", "pipe:0", "-show_entries", "frame=pict_type",
                         "-of", "csv=p=0"], input=enc, capture_output=True)
    types = [x.strip().strip(",") for x in pr.stdout.decode().split("\n") if x.strip()]
    dec = synth._run(["ffmpeg", "-v", "error", "-threads", "1", "-f", box, "-i", "-", "-fps_mode", "passthrough",
                      "-vf", "scale=in_color_matrix=bt709:in_range=tv,format=rgb24", "-f", "rawvideo", "-"], enc)
    out = np.frombuffer(dec, np.uint8).reshape(-1, hh, ww, 3)
    assert len(out) == n and len(types) == n, (len(out), len(types), codec, struct)
    mse = ((out.astype(np.float32) - clip.astype(np.float32)) ** 2).reshape(n, -1).mean(1)
    return job, out, types, 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-12))


def run_b(a):
    photos = ladder_rows()[:B_PHOTOS]
    jobs = [(r["sample_id"], r, ang, L, c, s, mo) for r, ang in photos for L in P.LADDER_L
            for c in B_CRF for s in B_STRUCT for mo in ("static", "pan")]
    done = set()
    if B_OUT.exists():
        with open(B_OUT, newline="", encoding="utf-8") as fh:
            done = {(r["photo"], float(r["L"]), r["codec"], r["struct"], r["motion"]) for r in csv.DictReader(fh)}
    todo = [j for j in jobs if (j[0], j[3], j[4], j[5], j[6]) not in done]
    C.log(f"(b) GOP-os létra: {len(photos)} fotó × {len(P.LADDER_L)} hossz × 3 kodek × 2 szerkezet × 2 mozgás = {len(jobs)} klip "
          f"({len(todo)} hátra), 8 frame; Update 3 + Small")
    u3, mu = C.load_u3()
    sm, _ = C.load_small()
    fh = open(B_OUT, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, B_COLS)
    if not done:
        w.writeheader()
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for n, (job, out, types, q) in enumerate(P.bounded_map(ex, _clip_job, todo, ahead=2 * a.workers)):
            x = torch.from_numpy(np.ascontiguousarray(out)).permute(0, 3, 1, 2).contiguous().cuda()
            lu = C.forward(u3, x)["image"].float().cpu().numpy()
            ls = C.forward(sm, x)["image"].float().cpu().numpy()
            sid, _, ang, L, codec, struct, motion = job
            for k in range(len(out)):
                w.writerow({"photo": sid, "L": L, "angle": ang, "codec": codec, "struct": struct, "motion": motion, "k": k,
                            "type": types[k], "psnr": f"{q[k]:.3f}", "u3_logit": f"{lu[k]:.6g}",
                            "u3_L": f"{float(C.measured_L(lu[k], mu)):.5g}", "small_logit": f"{ls[k]:.6g}"})
            if (n + 1) % 200 == 0:
                fh.flush()
                left = (time.time() - t0) / (n + 1) * (len(todo) - n - 1)
                C.log(f"  {n + 1} / {len(todo)} klip, hátra kb. {left / 60:.0f} perc")
    fh.close()
    C.log(f"kész: {B_OUT}")


def report(a):
    res = {}
    _, mu = C.load_u3("cpu")
    _, ms = C.load_small("cpu")
    metas = {"u3": mu, "small": ms}
    if A_OUT.exists():
        da, ds = np.load(A_OUT), np.load(S.OUT / "r4s1_ladder.npz")
        Ls = da["L"]
        low, mid = Ls <= 3, (Ls >= 3) & (Ls <= 16)
        ra = {"photos": int(da["u3"].shape[0]), "L_px": Ls.tolist(), "models": {}}
        for m, lg in (("u3", da["u3"]), ("small", ds["small"])):
            p = 1 / (1 + np.exp(-lg))
            per_L = [{"L": float(L), "median_logit": float(np.median(lg[:, j])),
                      **{f"blur_{k}": float(np.mean(p[:, j] >= t)) for k, t in metas[m]["thresholds"].items()}}
                     for j, L in enumerate(Ls)]
            e = {"per_L": per_L,
                 "flip_L_50pct": {k: E._flip(Ls, np.array([r[f"blur_{k}"] for r in per_L])) for k in metas[m]["thresholds"]},
                 "pair_order": {"0_3_px": E._pairs(lg, Ls, low), "3_16_px": E._pairs(lg, Ls, mid)},
                 "strictly_increasing_0_1_2_3": float(np.mean(np.all(np.diff(lg[:, [0, 2, 4, 6]], axis=1) > 0, axis=1)))}
            if m == "u3":
                Lm = C.measured_L(lg, mu)
                e["px"] = {"median_per_L": [float(np.median(Lm[:, j])) for j in range(len(Ls))],
                           "mae": float(np.abs(Lm - Ls).mean()), "mae_0_3": float(np.abs(Lm[:, low] - Ls[low]).mean()),
                           "spearman": float(spearmanr(np.broadcast_to(Ls, Lm.shape).ravel(), Lm.ravel())[0])}
            ra["models"][m] = e
        res["a_update1_ladder"] = ra
        for m, e in ra["models"].items():
            C.log(f"(a) {m}: átfordulás {e['flip_L_50pct']}, pár-sorrend 0-3 {e['pair_order']['0_3_px']['ordered']:.3f}, "
                  f"3-16 {e['pair_order']['3_16_px']['ordered']:.3f}, szigorú 0-1-2-3 {e['strictly_increasing_0_1_2_3']:.3f}")
        C.log(f"(a) u3 px: MAE {ra['models']['u3']['px']['mae']:.2f}, Spearman {ra['models']['u3']['px']['spearman']:.3f}, "
              f"medián L-enként {np.round(ra['models']['u3']['px']['median_per_L'], 2).tolist()}")
    if B_OUT.exists():
        with open(B_OUT, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        d = {c: np.array([r[c] for r in rows]) for c in ("photo", "codec", "struct", "motion", "type")}
        for c in ("L", "psnr", "u3_logit", "u3_L", "small_logit"):
            d[c] = np.array([float(r[c]) for r in rows])
        d["intra"] = d["type"] == "I"
        rb = {"frames": len(rows), "clips": len(rows) // 8, "photos": len(set(d["photo"])), "groups": {}}
        for codec in B_CRF:
            for st in B_STRUCT:
                for mo in ("static", "pan"):
                    for ft, fm in (("I", d["intra"]), ("P/B", ~d["intra"])):
                        g = (d["codec"] == codec) & (d["struct"] == st) & (d["motion"] == mo) & fm
                        if g.sum() < 50:
                            continue
                        z = g & (d["L"] == 0)
                        rb["groups"][f"{codec}/{st}/{mo}/{ft}"] = {
                            "n": int(g.sum()), "psnr_mean": float(d["psnr"][g].mean()),
                            "mae_px": float(np.abs(d["u3_L"][g] - d["L"][g]).mean()),
                            "mae_px_0_3": float(np.abs(d["u3_L"][g & (d["L"] <= 3)] - d["L"][g & (d["L"] <= 3)]).mean()),
                            "spearman_px_L": float(spearmanr(d["u3_L"][g], d["L"][g])[0]),
                            "median_px_at_0": float(np.median(d["u3_L"][z])),
                            "u3_blur_r95_at_0": float(np.mean(d["u3_logit"][z] >= C.logit_threshold(mu, "r95"))),
                            "small_blur_r95_at_0": float(np.mean(d["small_logit"][z] >= C.logit_threshold(ms, "r95"))),
                            "u3_median_logit_at_0": float(np.median(d["u3_logit"][z])),
                            "small_median_logit_at_0": float(np.median(d["small_logit"][z]))}
        res["b_gop_ladder"] = rb
        for k, e in rb["groups"].items():
            C.log(f"(b) {k:22s} n {e['n']:6d} PSNR {e['psnr_mean']:.1f} | MAE {e['mae_px']:.2f} (0-3: {e['mae_px_0_3']:.2f}) "
                  f"Spearman {e['spearman_px_L']:.3f} | 0 px: medián {e['median_px_at_0']:.2f} px, blur r95 U3 {e['u3_blur_r95_at_0']:.1%}"
                  f" Small {e['small_blur_r95_at_0']:.1%}")
    (C.DATA / "ladder.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("a", "b", "report"))
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    {"a": run_a, "b": run_b, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
