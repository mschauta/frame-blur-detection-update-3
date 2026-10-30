"""Protocol item 1: codec dependence on the fixed R4K-2 test (experiments/r4k2_gop_foto), unchanged clips.

The R4K-2 frames are not stored, only their scores, so every clip is coded again with R4K-2's own code (run.job: the
same photos, crops, x264 settings with -threads 1, rates and structures). Every re-coded frame must match the stored
row: frame type and packet size exactly, PSNR to 1e-3 dB. The re-coded frames are scored with Update 3 and, as a
check, again with the round3 Small; the comparison uses the Small's stored R4K-2 scores.

Measures (protocol): the paired logit difference between the P/B frames and the I-only coding of the same source
frame, and the difference in the share called blurred at r95, by rate and motion; the codec gap of Update 3 minus that
of the Small, with the same photo resamples (2000, as R4K-2: seed 20261028).

usage: python experiments/u3r3_eval/r4k2.py score [--threads 10]     (resumable)
       python experiments/u3r3_eval/r4k2.py report
Output: data/r4k2_u3.csv, data/r4k2.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import common as C

R4K2 = C.ROOT / "experiments" / "r4k2_gop_foto"
sys.path.insert(0, str(R4K2))
import run as R                                              # noqa: E402  R4K-2's own clips and coding
import analyse as A                                          # noqa: E402  R4K-2's bootstrap (photo multiplicities)

OUT = C.DATA / "r4k2_u3.csv"
COLS = ["photo", "motion", "rate", "struct", "k", "type", "pkt_size", "psnr", "u3_logit", "u3_L", "small_logit_rescored"]


def stored() -> dict:
    with open(R4K2 / "data" / "frames.csv", newline="", encoding="utf-8") as fh:
        return {(r["photo"], r["motion"], r["rate"], r["struct"], int(r["k"])): r for r in csv.DictReader(fh)}


def score(a):
    ref = stored()
    rows = R.photos(100)
    assert [r["sample_id"] for r in rows] == json.loads((R4K2 / "data" / "photos.json").read_text(encoding="utf-8"))
    jobs = [(r, m, rt, s) for r in rows for m in R.MOTIONS for rt in R.RATES for s in R.STRUCT]
    done = set()
    if OUT.exists():
        with open(OUT, newline="", encoding="utf-8") as fh:
            done = {(r["photo"], r["motion"], r["rate"], r["struct"]) for r in csv.DictReader(fh)}
    todo = [j for j in jobs if (j[0]["sample_id"], j[1], j[2], j[3]) not in done]
    C.log(f"R4K-2 újra: {len(jobs)} klip ({len(todo)} hátra) × {R.N} frame; Update 3 + round3 Small (ellenőrzés)")
    u3, mu = C.load_u3()
    sm, _ = C.load_small()
    fh = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, COLS)
    if not done:
        w.writeheader()
    worst = {"psnr": 0.0, "small_logit": 0.0}
    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as pool:
        ahead = [pool.submit(R.job, *j) for j in todo[:a.threads + 2]]
        for n in range(len(todo)):
            pid, motion, rate, struct, dec, types, sizes, q = ahead.pop(0).result()
            if n + a.threads + 2 < len(todo):
                ahead.append(pool.submit(R.job, *todo[n + a.threads + 2]))
            lu, ls = [], []
            for j in range(0, len(dec), 8):
                x = torch.from_numpy(np.ascontiguousarray(dec[j:j + 8])).permute(0, 3, 1, 2).contiguous().cuda()
                lu.append(C.forward(u3, x)["image"].float().cpu().numpy())
                ls.append(C.forward(sm, x)["image"].float().cpu().numpy())
            lu, ls = np.concatenate(lu), np.concatenate(ls)
            for k in range(R.N):
                s = ref[(pid, motion, rate, struct, k)]
                assert s["type"] == types[k] and int(s["pkt_size"]) == sizes[k], ("re-coding differs", pid, motion, rate, struct, k)
                worst["psnr"] = max(worst["psnr"], abs(float(s["psnr"]) - q[k]))
                worst["small_logit"] = max(worst["small_logit"], abs(float(s["small_logit"]) - ls[k]))
                w.writerow({"photo": pid, "motion": motion, "rate": rate, "struct": struct, "k": k, "type": types[k],
                            "pkt_size": sizes[k], "psnr": f"{q[k]:.3f}", "u3_logit": f"{lu[k]:.6g}",
                            "u3_L": f"{float(C.measured_L(lu[k], mu)):.5g}", "small_logit_rescored": f"{ls[k]:.6g}"})
            fh.flush()
            assert worst["psnr"] < 2e-3, worst
            if (n + 1) % 48 == 0:
                left = (time.time() - t0) / (n + 1) * (len(todo) - n - 1)
                C.log(f"  {n + 1} / {len(todo)} klip, hátra kb. {left / 60:.0f} perc; eltérés a tárolttól: PSNR {worst['psnr']:.1e} dB, "
                      f"Small logit {worst['small_logit']:.3f}")
    fh.close()
    C.log(f"kész; legnagyobb eltérés a tárolt R4K-2-sorokhoz: {worst}")


def report(a):
    ref = stored()
    with open(OUT, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(ref) == 115_200, (len(rows), len(ref))
    photos = sorted({r["photo"] for r in rows})
    pid = {p: i for i, p in enumerate(photos)}
    d = {"photo": np.array([pid[r["photo"]] for r in rows]), "k": np.array([int(r["k"]) for r in rows])}
    for c in ("motion", "rate", "struct", "type"):
        d[c] = np.array([r[c] for r in rows])
    d["u3"] = np.array([float(r["u3_logit"]) for r in rows])
    d["u3_L"] = np.array([float(r["u3_L"]) for r in rows])
    d["small"] = np.array([float(ref[(r["photo"], r["motion"], r["rate"], r["struct"], int(r["k"]))]["small_logit"]) for r in rows])
    d["small_rescored"] = np.array([float(r["small_logit_rescored"]) for r in rows])
    _, mu = C.load_u3("cpu")
    _, ms = C.load_small("cpu")
    th = {"u3": {p: C.logit_threshold(mu, p) for p in ("r95", "r98")}, "small": {p: C.logit_threshold(ms, p) for p in ("r95", "r98")}}
    B = A.Boot(len(photos))
    motions, rates = ("static", "pan"), ("crf23", "crf28", "crf33", "2mbps")
    key = lambda sel: list(zip(d["photo"][sel], d["motion"][sel], d["rate"][sel], d["k"][sel]))      # noqa: E731
    ref_sel = d["struct"] == "I"
    iref = {k: i for k, i in zip(key(ref_sel), np.flatnonzero(ref_sel))}
    res = {"photos": len(photos), "frames": len(rows), "thresholds_logit": th,
           "check_small_rescored_minus_stored": {"max_abs": float(np.abs(d["small_rescored"] - d["small"]).max()),
                                                 "median_abs": float(np.median(np.abs(d["small_rescored"] - d["small"])))},
           "bootstrap": f"{A.REPS} resamples over the photos, seed {A.SEED} (R4K-2's)"}
    # false alarms (every frame is 0 px sharp)
    res["false_alarms_total"] = {f"{m}/{p}": {"blurred": int((d[m] >= t).sum()), "of": len(rows)}
                                 for m in ("u3", "small") for p, t in th[m].items()}
    # paired against the I-only coding, by structure and type (as R4K-2), and pooled over the P/B frames
    paired, pooled = {}, {}
    pb = (d["struct"] != "I") & (d["type"] != "I")
    for st in ("IP", "IPB"):
        sel = np.flatnonzero(d["struct"] == st)
        j = np.array([iref[k] for k in key(d["struct"] == st)])
        for mo in motions:
            for ra in rates:
                for ty in ("I", "P", "B"):
                    g = (d["motion"][sel] == mo) & (d["rate"][sel] == ra) & (d["type"][sel] == ty)
                    if not g.any():
                        continue
                    p = d["photo"][sel][g]
                    e = {}
                    for m in ("u3", "small"):
                        diff = d[m][sel][g] - d[m][j][g]
                        e[m] = {"score_difference": B.mean(p, diff)}
                        for pt, t in th[m].items():
                            e[m][f"blurred_share_difference_{pt}"] = B.mean(p, (d[m][sel][g] >= t).astype(float) - (d[m][j][g] >= t))
                    paired[f"{mo}/{ra}/{st}/{ty}"] = e
    # pooled P/B gap per motion x rate, and U3 minus Small with the same resamples
    allj = np.full(len(rows), -1)
    for st in ("IP", "IPB"):
        sel = np.flatnonzero(d["struct"] == st)
        allj[sel] = [iref[k] for k in key(d["struct"] == st)]
    for mo in motions:
        for ra in rates:
            g = np.flatnonzero(pb & (d["motion"] == mo) & (d["rate"] == ra))
            jj, p = allj[g], d["photo"][g]
            e = {}
            for m in ("u3", "small"):
                e[f"{m}_logit_gap"] = B.mean(p, d[m][g] - d[m][jj])
                e[f"{m}_blur_gap_r95"] = B.mean(p, (d[m][g] >= th[m]["r95"]).astype(float) - (d[m][jj] >= th[m]["r95"]))
            e["u3_minus_small_logit_gap"] = B.diff(p, d["u3"][g] - d["u3"][jj], p, d["small"][g] - d["small"][jj])
            bu = (d["u3"][g] >= th["u3"]["r95"]).astype(float) - (d["u3"][jj] >= th["u3"]["r95"])
            bs = (d["small"][g] >= th["small"]["r95"]).astype(float) - (d["small"][jj] >= th["small"]["r95"])
            e["u3_minus_small_blur_gap_r95"] = B.diff(p, bu, p, bs)
            # the ratio of the gaps, bootstrapped with the same photo multiplicities
            su, ss, cnt = (np.bincount(p, v, B.n) for v in (d["u3"][g] - d["u3"][jj], d["small"][g] - d["small"][jj], None))
            ratio = (B.W @ su) / np.where(np.abs(B.W @ ss) > 1e-12, B.W @ ss, np.nan)
            e["u3_over_small_logit_gap"] = {"value": float(su.sum() / ss.sum()), "ci95": np.nanpercentile(ratio, [2.5, 97.5]).tolist()}
            e["u3_px_P_B_minus_I_only"] = B.mean(p, d["u3_L"][g] - d["u3_L"][jj])
            e["u3_px_P_B_median"] = float(np.median(d["u3_L"][g]))
            e["u3_px_I_only_median"] = float(np.median(d["u3_L"][jj]))
            pooled[f"{mo}/{ra}"] = e
    res["paired_vs_I_only"] = paired
    res["pooled_P_B_gap"] = pooled
    # R4K-1 repeated: the in-clip I-frame against its two neighbours (IPB, protocol of R4K-2)
    idx = {k: i for k, i in zip(zip(d["photo"], d["motion"], d["rate"], d["struct"], d["k"]), range(len(rows)))}
    rep = {}
    for st in ("IP", "IPB"):
        for m in ("u3", "small"):
            for mo in motions:
                for ra in rates:
                    pg, lg, pn, ln = [], [], [], []
                    for ph in range(len(photos)):
                        for k in (15, 30, 45):
                            g = [idx[(ph, mo, ra, st, kk)] for kk in (k - 1, k, k + 1)]
                            nn = [idx[(ph, mo, ra, "I", kk)] for kk in (k - 1, k, k + 1)]
                            v, u = d[m][g], d[m][nn]
                            pg.append(ph); lg.append(float(v[1] < min(v[0], v[2])))
                            pn.append(ph); ln.append(float(u[1] < min(u[0], u[2])))
                    pg, lg, pn, ln = map(np.array, (pg, lg, pn, ln))
                    rep[f"{m}/{mo}/{ra}/{st}"] = {"i_frame_sharpest": B.mean(pg, lg), "null": B.mean(pn, ln)}
    res["r4k1_repeat"] = rep
    (C.DATA / "r4k2.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    C.log(f"ellenőrzés: Small újrapontozva − tárolt: {res['check_small_rescored_minus_stored']}")
    C.log(f"téves riasztás: {json.dumps(res['false_alarms_total'])}")
    for k, e in pooled.items():
        C.log(f"  {k:12s} logit-rés U3 {e['u3_logit_gap']['value']:+6.2f}, Small {e['small_logit_gap']['value']:+6.2f}, "
              f"arány {e['u3_over_small_logit_gap']['value']:.2f} {np.round(e['u3_over_small_logit_gap']['ci95'], 2).tolist()} | "
              f"blur-rés r95 U3 {e['u3_blur_gap_r95']['value']:+.3f}, Small {e['small_blur_gap_r95']['value']:+.3f}, "
              f"különbség {e['u3_minus_small_blur_gap_r95']['value']:+.3f} {np.round(e['u3_minus_small_blur_gap_r95']['ci95'], 3).tolist()}"
              f" | U3 px P/B {e['u3_px_P_B_median']:.2f} vs I {e['u3_px_I_only_median']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("score", "report"))
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()
    {"score": score, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
