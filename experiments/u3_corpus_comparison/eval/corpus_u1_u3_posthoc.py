"""Added after the results of corpus_u1_u3_report.py (post hoc, not in the plan of napló 2026-11-01 23:35):

cross: the Qwen codes in each Update 1 tier x Update 3 tier cell (do the two models together say more about tier C?).
films: the films where the two models differ most, by number only.
sample: 300,000 frames drawn at random (seed 20261030), from the stored maps:
    the frame score without the red cells: does the frame stay above r98 on the field below logit 0 alone?
    where the red cells sit: the percent rank of measured motion, inside the frame, of the 160 px blocks holding them.

usage: python experiments/u3r3_eval/corpus_u1_u3_posthoc.py --cache DIR [--procs 16]
Output: data/corpus_u1_u3_posthoc.json. Film names are not written.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import zlib
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

import corpus_u1_u3 as X
from corpus_u1_u3_report import CODES, EXTRA, boot, pct, rnd

N_SAMPLE = 300_000


def nonred(v, r):
    """Log-mean-exp over all cells, with the red cells contributing nothing, and the share of the sum carried by red."""
    s = v.astype(np.float64) / 4 * r
    top = s.max()
    e = np.exp(s - top)
    total = e.sum()
    below = e[v < 0].sum()
    return ((top + np.log(below / v.size)) / r if below > 0 else -np.inf), 1 - below / total


def red_rank(m, cells, ranks):
    """Red cells of the map inside the measured blocks: their mean percent rank of block motion; red cells covered."""
    weight = np.array([(m[a1:b1, a0:b0] >= 0).sum() for a0, a1, b0, b1 in cells], float)
    return (weight * ranks).sum() / weight.sum() if weight.sum() > 0 else np.nan


def work(job):
    idx, fids, rid1, rid3 = job
    con = X.connect()
    rows = {}
    ask = np.concatenate([rid1, rid3]).tolist()
    for start in range(0, len(ask), 2000):
        part = ",".join(map(str, ask[start:start + 2000]))
        for rid, r, c, p in con.execute(f"select rowid, rows, cols, payload from frame_blur_cells where rowid in ({part})"):
            rows[rid] = (r, c, p)
    evid = {}
    for start in range(0, len(fids), 500):
        part = fids[start:start + 500]
        for fid, p in con.execute(f"select frame_id, payload from frame_measurement_evidence where frame_id in ({','.join('?' * len(part))})", part):
            evid[fid] = p
    con.close()
    # per frame, per model: score without red, share carried by red, red motion rank, red cells in the measured blocks; then the same cells of the other model
    out = np.full((len(fids), 6), np.nan, np.float32)
    for i, fid in enumerate(fids):
        r1, c1, p1 = rows[int(rid1[i])]
        r3, c3, p3 = rows[int(rid3[i])]
        a = np.frombuffer(zlib.decompress(p1), np.int8)
        b = np.frombuffer(zlib.decompress(p3), np.int8)
        out[i, 0], out[i, 1] = nonred(a, X.R["u1"])
        out[i, 2], out[i, 3] = nonred(b, X.R["u3"])
        ev = evid.get(fid)
        if ev is None:
            continue
        mc = json.loads(zlib.decompress(ev))["motion_cells"]
        if mc["compensated"].get("status") != "measured":
            continue
        cells, travel = [], []
        for c in mc["compensated"]["cells"]:
            if c["status"] != "measured" or c.get("structured_count", 0) < 500 or c.get("travel_px") is None:
                continue
            x0, y0, x1, y1 = c["bbox"]
            cells.append((x0 // 16, y0 // 16, -(-x1 // 16), -(-y1 // 16)))
            travel.append(c["travel_px"])
        if len(cells) < 8:
            continue
        ranks = (rankdata(travel) - 0.5) / len(travel)
        out[i, 4] = red_rank(a.reshape(r1, c1), cells, ranks)
        out[i, 5] = red_rank(b.reshape(r3, c3), cells, ranks)
    return idx, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    cache = Path(a.cache)
    ids = np.load(cache / "ids.npy")
    facts = dict(np.load(cache / "facts.npz"))
    names = json.loads((cache / "names.json").read_text(encoding="utf-8"))
    feats = np.load(cache / "features.npy", mmap_mode="r")
    col = {n: i for i, n in enumerate(json.loads((cache / "features.json").read_text(encoding="utf-8")))}
    F = lambda n: np.asarray(feats[:, col[n]])       # noqa: E731
    N = len(ids)
    L = {"u1": facts["logit_u1"], "u3": facts["logit_u3"]}
    th = {m: {p: float(np.log(t / (1 - t))) for p, t in names["thresholds"][m].items()} for m in X.SETS}
    tier = {m: np.digitize(L[m], [th[m]["r98"], th[m]["r95"], th[m]["r90"]]).astype(np.int8) for m in L}
    video, scene, code = facts["video"], facts["scene"], facts["code"]
    films = int(video.max()) + 1
    kept = facts["selected"] & ~facts["teacher_branch"]
    shot_kept = np.bincount(scene[scene >= 0], kept[scene >= 0], minlength=int(scene.max()) + 1)
    zero_shot = (scene >= 0) & (shot_kept[np.maximum(scene, 0)] == 0)
    res = {}

    # cross: the codes in each tier pair
    def cell(m):
        lab = m & (code >= 0)
        return {"frames": int(m.sum()), "labelled": int(lab.sum()), "sss": pct((code[lab] == 0).mean()) if lab.any() else None,
                "blurred_codes": pct(np.isin(code[lab], (3, 5)).mean()) if lab.any() else None, "bbb": pct((code[lab] == 5).mean()) if lab.any() else None}
    res["cross_codes"] = {}
    for g, gm in (("all", np.ones(N, bool)), ("zero_shots", zero_shot), ("other_shots", ~zero_shot)):
        res["cross_codes"][g] = {f"{'ABCD'[i]}{'ABCD'[j]}": cell(gm & (tier["u1"] == i) & (tier["u3"] == j)) for i in range(4) for j in range(4)}
    X.log("cross")

    # maps: overlap where both models have red
    both = F("red_both"), F("red_either")
    r1, r3 = F("u1_red") > 0, F("u3_red") > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = both[0] / both[1]
    res["iou_both_have_red"] = {}
    for name, gm in {"all": np.ones(N, bool), "both_A": (tier["u1"] == 0) & (tier["u3"] == 0), "both_B_C": np.isin(tier["u1"], (1, 2)) & np.isin(tier["u3"], (1, 2)), "both_D": (tier["u1"] == 3) & (tier["u3"] == 3)}.items():
        m = gm & r1 & r3
        res["iou_both_have_red"][name] = {"frames": int(m.sum()), "iou_median": rnd(np.nanmedian(iou[m])), "iou_p25_p75": [rnd(q) for q in np.nanpercentile(iou[m], [25, 75])], "no_overlap_share": pct((both[0][m] == 0).mean()),
                                          "u1_yellow_median": pct(np.median(F("u1_yellow")[m])), "u3_yellow_median": pct(np.median(F("u3_yellow")[m]))}
    res["yellow_median"] = {T: {m: pct(np.median(F(f"{m}_yellow")[tier[m] == i])) for m in L} for i, T in enumerate("ABCD")}
    res["cell_median_by_tier"] = {T: {m: rnd(np.median(F(f"{m}_median")[tier[m] == i]), 2) for m in L} for i, T in enumerate("ABCD")}
    X.log(f"iou {res['iou_both_have_red']}")

    # films that differ most
    extra = dict(np.load(cache / "extra.npz"))
    rows = []
    con = X.connect()
    vf = {v: (s, d) for v, s, d in con.execute("select id, source_size_bytes, duration_seconds from videos")}
    for f in range(films):
        at = np.flatnonzero(video == f)
        s, d = vf[names["videos"][f]]
        rows.append({"film": f, "frames": int(len(at)), "mbps": rnd(s * 8 / d / 1e6, 2), "spearman": rnd(spearmanr(L["u1"][at], L["u3"][at])[0]),
                     **{f"sharp_{p}": [pct((L[m][at] < th[m][p]).mean()) for m in L] for p in ("r95", "r98")},
                     "lum": rnd(np.nanmedian(extra["lum"][at]), 3), "noise": rnd(np.nanmedian(extra["noise"][at]), 3), "block": rnd(np.nanmedian(extra["block"][at]), 3), "lap": rnd(np.nanmedian(extra["lap"][at]), 1)})
    rows.sort(key=lambda r: r["spearman"])
    res["films_lowest_spearman"] = rows[:6]
    res["films_largest_r98_shift"] = sorted(rows, key=lambda r: r["sharp_r98"][1] - r["sharp_r98"][0])[:6]
    res["film_medians"] = {k: rnd(np.median([r[k] for r in rows]), 3) for k in ("lum", "noise", "block", "lap", "mbps", "spearman")}
    res["films_spearman_vs"] = {k: rnd(spearmanr([r["spearman"] for r in rows], [r[k] for r in rows])[0]) for k in ("lum", "noise", "block", "lap", "mbps", "frames")}
    X.log(f"films {res['films_lowest_spearman']}")

    # sample pass
    rng = np.random.default_rng(X.SEED)
    pick = np.sort(rng.choice(N, N_SAMPLE, replace=False))
    cur = con.execute("select frame_id, set_id, rowid from frame_blur_cells indexed by sqlite_autoindex_frame_blur_cells_1")
    rid = {m: np.zeros(N, np.int64) for m in X.SETS}
    name_of = {v: k for k, v in X.SETS.items()}
    while True:
        part = cur.fetchmany(1_000_000)
        if not part:
            break
        at = np.searchsorted(ids, np.array([p[0] for p in part], "S36"))
        for m in X.SETS:
            sel = np.array([name_of.get(p[1]) == m for p in part])
            rid[m][at[sel]] = np.array([p[2] for p, ok in zip(part, sel) if ok], np.int64)
    con.close()
    walk = pick[np.argsort(rid["u1"][pick])]
    jobs = [(walk[s:s + 3000], [x.decode() for x in ids[walk[s:s + 3000]]], rid["u1"][walk[s:s + 3000]], rid["u3"][walk[s:s + 3000]]) for s in range(0, len(walk), 3000)]
    out = np.full((N, 6), np.nan, np.float32)
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for idx, o in pool.imap_unordered(work, jobs):
            out[idx] = o
    X.log("sample done")
    s = np.zeros(N, bool)
    s[pick] = True
    res["sample_frames"] = N_SAMPLE
    res["without_red"] = {}
    for g, gm in (("all", s), ("zero_shots", s & zero_shot), ("other_shots", s & ~zero_shot)):
        e = {}
        for j, m in enumerate(L):
            nr, share = out[:, 2 * j], out[:, 2 * j + 1]
            e[m] = {}
            for i, T in enumerate("BCD"):
                k = gm & (tier[m] == i + 1)
                lab = k & (code == 0)
                e[m][T] = {"frames": int(k.sum()), "stays_above_r98_without_red": pct((nr[k] >= th[m]["r98"]).mean()), "red_carries_median": pct(np.nanmedian(share[k])),
                           "sss_frames": int(lab.sum()), "sss_stays_above_r98_without_red": pct((nr[lab] >= th[m]["r98"]).mean()) if lab.any() else None}
        res["without_red"][g] = e
    res["red_motion_rank"] = {}
    for g, tiers in (("A", (0,)), ("B", (1,)), ("C", (2,)), ("D", (3,))):
        e = {}
        for j, m in enumerate(L):
            k = s & np.isin(tier[m], tiers) & np.isfinite(out[:, 4 + j])
            e[m] = {"frames": int(k.sum()), "median": rnd(np.median(out[k, 4 + j])), "mean": rnd(np.mean(out[k, 4 + j])), "top_half": pct((out[k, 4 + j] > 0.5).mean())}
        res["red_motion_rank"][g] = e
    # on the same frames: every model's own red, both models sharp or both blurred
    for name, both_m in (("both_A", (tier["u1"] == 0) & (tier["u3"] == 0)), ("both_D", (tier["u1"] == 3) & (tier["u3"] == 3))):
        k = s & both_m & np.isfinite(out[:, 4]) & np.isfinite(out[:, 5])
        d = np.full(films, np.nan)
        for f in range(films):
            at = k & (video == f)
            if at.sum() >= 30:
                d[f] = np.median(out[at, 5]) - np.median(out[at, 4])
        res["red_motion_rank"][name] = {"frames": int(k.sum()), "u1_median": rnd(np.median(out[k, 4])), "u3_median": rnd(np.median(out[k, 5])), "u3_minus_u1_films": boot(d)}
    X.log(f"without red {res['without_red']}")
    X.log(f"red motion {res['red_motion_rank']}")
    (X.DATA / "corpus_u1_u3_posthoc.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res["cross_codes"], ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("iou_both_have_red", "yellow_median", "cell_median_by_tier", "films_lowest_spearman", "films_largest_r98_shift", "film_medians", "films_spearman_vs")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
