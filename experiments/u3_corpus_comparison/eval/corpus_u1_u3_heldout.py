"""The part of the plan (napló 2026-11-01 23:35) left out of corpus_u1_u3_report.py: the 6 held-out films, per film and with a
bootstrap over their shots (2000, seed 20261030). u3 - u1; "eq" = per film, u3 keeps as many frames as u1 at that point.

usage: python experiments/u3r3_eval/corpus_u1_u3_heldout.py --cache DIR
Output: data/corpus_u1_u3_heldout.json. Film names are not written.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

import corpus_u1_u3 as X
from corpus_u1_u3_report import MIN_SHOT, corr, rnd

BOOT = 2000


def ci(values, rng):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    meds = np.median(v[rng.integers(len(v), size=(BOOT, len(v)))], axis=1)
    return {"median": rnd(np.median(v), 4), "ci95": [rnd(q, 4) for q in np.percentile(meds, [2.5, 97.5])], "shots": int(len(v))}


def auc(score, y):
    r = rankdata(score)
    pos = y == 1
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    a = ap.parse_args()
    cache = Path(a.cache)
    ids = np.load(cache / "ids.npy")
    facts = dict(np.load(cache / "facts.npz"))
    names = json.loads((cache / "names.json").read_text(encoding="utf-8"))
    feats = np.load(cache / "features.npy", mmap_mode="r")
    col = {n: i for i, n in enumerate(json.loads((cache / "features.json").read_text(encoding="utf-8")))}
    L = {"u1": facts["logit_u1"], "u3": facts["logit_u3"]}
    th = {m: {p: float(np.log(t / (1 - t))) for p, t in names["thresholds"][m].items()} for m in X.SETS}
    video, scene, fidx, region = facts["video"], facts["scene"], facts["fidx"], facts["region"]
    wit = {"u1": np.asarray(feats[:, col["wit_comp_u1"]]), "u3": np.asarray(feats[:, col["wit_comp_u3"]])}
    target = np.full(len(ids), -1, np.int8)
    with open(X.INDEX, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["source"] == "frames" and r["split"] in ("val", "test")]
    at = np.searchsorted(ids, np.array([r["sample_id"].split("/", 1)[1] for r in rows], "S36"))
    target[at] = [int(r["target"]) for r in rows]
    held = np.unique(video[target >= 0])
    res = {"films": int(len(held)), "per_film": [], "points": {}}
    rng = np.random.default_rng(X.SEED)
    shots = []          # (film, spearman u1, spearman u3, witness u1, witness u3, per point: flips u1, flips u3 eq, pairs)
    POINTS = ("r90", "r95", "r98")
    for f in held:
        fa = np.flatnonzero(video == f)
        eq = {}
        for p in POINTS:
            n = int((L["u1"][fa] < th["u1"][p]).sum())
            k = np.zeros(len(fa), bool)
            k[np.argpartition(L["u3"][fa], n - 1)[:n]] = True
            eq[p] = k
        order = np.lexsort((fidx[fa], scene[fa]))
        sc = scene[fa][order]
        edges = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1], True])
        for s, e in zip(edges[:-1], edges[1:]):
            if e - s < MIN_SHOT:
                continue
            o = order[s:e]
            i = fa[o]
            ok = np.isfinite(region[i])
            if ok.sum() < MIN_SHOT:
                continue
            rx = rankdata(region[i][ok])
            row = [int(f), corr(rankdata(L["u1"][i][ok]), rx), corr(rankdata(L["u3"][i][ok]), rx), np.nanmedian(wit["u1"][i]), np.nanmedian(wit["u3"][i])]
            for p in POINTS:
                k1 = L["u1"][i] < th["u1"][p]
                k3 = eq[p][o]
                row += [(k1[1:] != k1[:-1]).sum(), (k3[1:] != k3[:-1]).sum()]
            row.append(e - s - 1)
            shots.append(row)
    S = np.array(shots, float)
    n_shots = len(S)
    X.log(f"held-out shots {n_shots}")
    res["shots"] = n_shots
    res["within_shot_logit_motion"] = {"u1": rnd(np.nanmedian(S[:, 1])), "u3": rnd(np.nanmedian(S[:, 2])), "u3_minus_u1": ci(S[:, 2] - S[:, 1], rng)}
    res["cell_witness"] = {"u1": rnd(np.nanmedian(S[:, 3])), "u3": rnd(np.nanmedian(S[:, 4])), "u3_minus_u1": ci(S[:, 4] - S[:, 3], rng)}
    pairs = S[:, -1]
    for j, p in enumerate(POINTS):
        f1, f3 = S[:, 5 + 2 * j], S[:, 6 + 2 * j]
        pick = rng.integers(n_shots, size=(BOOT, n_shots))
        d = (f3[pick].sum(1) - f1[pick].sum(1)) / pairs[pick].sum(1) * 1000
        res["points"][p] = {"flips_per_1000": {"u1": rnd(f1.sum() / pairs.sum() * 1000, 2), "u3_eq": rnd(f3.sum() / pairs.sum() * 1000, 2)}, "u3_minus_u1": rnd((f3.sum() - f1.sum()) / pairs.sum() * 1000, 2), "ci95": [rnd(q, 2) for q in np.percentile(d, [2.5, 97.5])]}
    # AUC, paired, over the shots of the labelled index frames
    lab = np.flatnonzero(target >= 0)
    shot_of = scene[lab]
    uniq, inv = np.unique(shot_of, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    a1, a3 = auc(L["u1"][lab], target[lab]), auc(L["u3"][lab], target[lab])
    diffs = []
    for _ in range(BOOT):
        take = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        y = target[lab][take]
        diffs.append(auc(L["u3"][lab][take], y) - auc(L["u1"][lab][take], y))
    res["auc_val_test"] = {"u1": rnd(a1, 5), "u3": rnd(a3, 5), "u3_minus_u1": rnd(a3 - a1, 5), "ci95": [rnd(q, 5) for q in np.percentile(diffs, [2.5, 97.5])], "frames": int(len(lab)), "shots": int(len(groups))}
    for n, f in enumerate(held):
        fa = np.flatnonzero(video == f)
        s = S[S[:, 0] == f]
        e = {"film": n, "frames": int(len(fa)), "shots": int(len(s)), "sharp_r98": [rnd((L[m][fa] < th[m]["r98"]).mean() * 100, 1) for m in L], "sharp_r95": [rnd((L[m][fa] < th[m]["r95"]).mean() * 100, 1) for m in L],
             "logit_motion": [rnd(np.nanmedian(s[:, 1])), rnd(np.nanmedian(s[:, 2]))], "cell_witness": [rnd(np.nanmedian(s[:, 3])), rnd(np.nanmedian(s[:, 4]))], "flips_r95": [rnd(s[:, 7].sum() / s[:, -1].sum() * 1000, 1), rnd(s[:, 8].sum() / s[:, -1].sum() * 1000, 1)]}
        lb = fa[target[fa] >= 0]
        e["auc"] = [rnd(auc(L[m][lb], target[lb]), 4) for m in L]
        res["per_film"].append(e)
    (X.DATA / "corpus_u1_u3_heldout.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
