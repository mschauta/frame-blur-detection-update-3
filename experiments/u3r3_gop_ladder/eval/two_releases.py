"""Added to item 5 (napló 2026-10-30, after the first episode): the 31 corpus films released twice (Update 1 README).

The test series gave Update 3 a poor agreement between its AVC and HEVC encodes. Is Update 3 unstable across encodes
in general, or only on that series? Here the same film in two H.264 releases at different bitrates (both 8 bit,
1920 x 1080, 5-12 Mbit/s; selection_rules_v8 same_film). Frames matched by time, the offset estimated from the shot
starts (tools/evaluate_scores.py, section 8), at most 400 matched pairs per film drawn at random (seed 20261030),
scored with Update 3 and the round3 Small through the published Update 1 loader. These films are in the corpus (most
in the training split), so both models may have seen frames of them.

Measures per film: Spearman and Pearson of the logits between the two releases, the r95 agreement and kappa, the
median logit of the lower-bitrate copy minus the higher; medians over films and the U3 - Small difference with a
bootstrap over films.

usage: python experiments/u3r3_eval/two_releases.py score
       python experiments/u3r3_eval/two_releases.py report
Output: data/two_releases.csv, data/two_releases.json (film names are not written)
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr

import common as C

RULES = Path("F:/RGB Mesh/dataset/frames/selection_rules_v8.json")
MASTER = Path("<master>")
OUT = C.DATA / "two_releases.csv"
PER_FILM = 400


def pairs():
    rules = json.loads(RULES.read_text(encoding="utf-8"))
    con = sqlite3.connect(f"file:{MASTER / 'master.db'}?mode=ro", uri=True)
    vids = {}
    for vid, path, size, dur, fps in con.execute("SELECT id, source_path, source_size_bytes, duration_seconds, average_fps FROM videos"):
        vids[path.replace("\\", "/").split("/")[-1]] = (vid, size * 8 / dur / 1e6, fps or 25.0)
    starts = defaultdict(list)
    for vid, st in con.execute("SELECT video_id, start_time FROM scenes"):
        starts[vid].append(st)
    rng = np.random.default_rng(C.SEED)
    out = []
    for n, (copy, kept) in enumerate(sorted(rules["same_film"].items())):
        if kept not in vids or copy not in vids:
            continue
        (va, ba, fps), (vb, bb, _) = vids[kept], vids[copy]
        A, B = np.sort(starts[va]), np.sort(starts[vb])
        best, off = -1.0, 0.0
        for o in np.arange(-3, 3.001, 0.02):
            j = np.searchsorted(B, A + o).clip(1, len(B) - 1)
            fr = float(np.mean(np.minimum(abs(B[j] - (A + o)), abs(B[j - 1] - (A + o))) < 0.15))
            if fr > best:
                best, off = fr, o
        fa = con.execute("SELECT id, timestamp_seconds, rgb_path FROM frames WHERE video_id = ? ORDER BY timestamp_seconds", (va,)).fetchall()
        fb = con.execute("SELECT id, timestamp_seconds, rgb_path FROM frames WHERE video_id = ? ORDER BY timestamp_seconds", (vb,)).fetchall()
        tb = np.array([r[1] for r in fb])
        ta = np.array([r[1] for r in fa]) + off
        j = np.searchsorted(tb, ta).clip(1, len(tb) - 1)
        jj = np.where(abs(tb[j] - ta) < abs(tb[j - 1] - ta), j, j - 1)
        ok = np.flatnonzero(abs(tb[jj] - ta) < 0.5 / fps)
        pick = np.sort(rng.choice(ok, min(PER_FILM, len(ok)), replace=False))
        lo_is_a = ba < bb
        for i in pick:
            a, b = fa[i], fb[jj[i]]
            low, high = (a, b) if lo_is_a else (b, a)
            out.append({"film": n, "shot_match": best, "offset_s": off, "mbps_low": min(ba, bb), "mbps_high": max(ba, bb),
                        "low_id": low[0], "high_id": high[0], "low_png": low[2], "high_png": high[2]})
    return out


def score(a):
    rows = pairs()
    done = set()
    if OUT.exists():
        with open(OUT, newline="", encoding="utf-8") as fh:
            done = {(r["low_id"], r["high_id"]) for r in csv.DictReader(fh)}
    todo = [r for r in rows if (r["low_id"], r["high_id"]) not in done]
    C.log(f"két kiadás: {len(set(r['film'] for r in rows))} film, {len(rows)} pár, hátra {len(todo)}")
    u3, mu = C.load_u3()
    sm, _ = C.load_small()
    load = lambda p: np.asarray(Image.open(MASTER / p).convert("RGB"))       # noqa: E731
    cols = ["film", "shot_match", "offset_s", "mbps_low", "mbps_high", "low_id", "high_id",
            "u3_low", "u3_high", "small_low", "small_high"]
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as fh, ThreadPoolExecutor(6) as pool:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        if new:
            w.writeheader()
        ahead = [pool.submit(lambda r: (load(r["low_png"]), load(r["high_png"])), r) for r in todo[:12]]
        for n, r in enumerate(todo):
            lo, hi = ahead.pop(0).result()
            if n + 12 < len(todo):
                ahead.append(pool.submit(lambda r: (load(r["low_png"]), load(r["high_png"])), todo[n + 12]))
            res = {}
            for tag, img in (("low", lo), ("high", hi)):
                x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].cuda()
                res[f"u3_{tag}"] = float(C.forward(u3, x)["image"].float())
                res[f"small_{tag}"] = float(C.forward(sm, x)["image"].float())
            w.writerow({**r, **{k: f"{v:.6g}" for k, v in res.items()}})
            if (n + 1) % 1000 == 0:
                fh.flush()
                C.log(f"  {n + 1} / {len(todo)}")
    C.log(f"kész: {OUT}")


def report(a):
    with open(OUT, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    _, mu = C.load_u3("cpu")
    _, ms = C.load_small("cpu")
    th = {"u3": C.logit_threshold(mu), "small": C.logit_threshold(ms)}
    films = sorted({int(r["film"]) for r in rows})
    per = []
    for f in films:
        g = [r for r in rows if int(r["film"]) == f]
        e = {"pairs": len(g), "mbps_low": float(g[0]["mbps_low"]), "mbps_high": float(g[0]["mbps_high"]),
             "shot_match": float(g[0]["shot_match"])}
        for m in ("u3", "small"):
            lo, hi = np.array([float(r[f"{m}_low"]) for r in g]), np.array([float(r[f"{m}_high"]) for r in g])
            dl, dh = lo >= th[m], hi >= th[m]
            po = dl.mean() * dh.mean() + (1 - dl.mean()) * (1 - dh.mean())
            agree = float(np.mean(dl == dh))
            e[m] = {"spearman": float(spearmanr(lo, hi)[0]), "pearson": float(pearsonr(lo, hi)[0]), "agree_r95": agree,
                    "kappa_r95": (agree - po) / (1 - po) if po < 1 else float("nan"),
                    "median_low_minus_high": float(np.median(lo - hi)), "blurred_low": float(dl.mean()), "blurred_high": float(dh.mean())}
        e["u3_px_median_low"] = float(np.median(C.measured_L(np.array([float(r["u3_low"]) for r in g]), mu)))
        e["u3_px_median_high"] = float(np.median(C.measured_L(np.array([float(r["u3_high"]) for r in g]), mu)))
        per.append(e)
    rng = np.random.default_rng(C.SEED)
    res = {"films": len(per), "pairs": len(rows), "per_film": per, "medians": {}}
    for k in ("spearman", "pearson", "agree_r95", "kappa_r95", "median_low_minus_high"):
        u = np.array([e["u3"][k] for e in per]); s = np.array([e["small"][k] for e in per])
        boot = [np.median(u[b] - s[b]) for b in (rng.integers(len(u), size=len(u)) for _ in range(2000))]
        res["medians"][k] = {"u3": float(np.median(u)), "small": float(np.median(s)), "u3_minus_small_median": float(np.median(u - s)),
                             "ci95": np.percentile(boot, [2.5, 97.5]).tolist(), "u3_better_films": int((u > s).sum())}
        C.log(f"{k}: U3 {np.median(u):.3f}, Small {np.median(s):.3f}; U3 − Small medián {np.median(u - s):+.3f} "
              f"{np.round(np.percentile(boot, [2.5, 97.5]), 3).tolist()}; U3 nagyobb {int((u > s).sum())} / {len(u)} filmben")
    (C.DATA / "two_releases.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("score", "report"))
    a = ap.parse_args()
    {"score": score, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
