"""Post hoc (after corpus_u1_u3_posthoc.py): the two largest r98 shifts are among the three films of highest bitrate. Every film above
12 Mbps against the rest, with the Qwen codes as a gross check and the motion as a witness. Read-only; film names are not written.

usage: python experiments/u3r3_eval/corpus_u1_u3_bitrate.py --cache DIR
Output: data/corpus_u1_u3_bitrate.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import corpus_u1_u3 as X
from corpus_u1_u3_report import pct, rnd

LIMIT = 12.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    a = ap.parse_args()
    cache = Path(a.cache)
    facts = dict(np.load(cache / "facts.npz"))
    extra = dict(np.load(cache / "extra.npz"))
    names = json.loads((cache / "names.json").read_text(encoding="utf-8"))
    feats = np.load(cache / "features.npy", mmap_mode="r")
    col = {n: i for i, n in enumerate(json.loads((cache / "features.json").read_text(encoding="utf-8")))}
    F = lambda n: np.asarray(feats[:, col[n]])       # noqa: E731
    L = {"u1": facts["logit_u1"], "u3": facts["logit_u3"]}
    th = {m: {p: float(np.log(t / (1 - t))) for p, t in names["thresholds"][m].items()} for m in X.SETS}
    video, code, region = facts["video"], facts["code"], facts["region"]
    con = X.connect()
    vf = {v: (s, d, w, h, c) for v, s, d, w, h, c in con.execute("select id, source_size_bytes, duration_seconds, width, height, codec from videos")}
    films = int(video.max()) + 1
    mbps = np.array([vf[v][0] * 8 / vf[v][1] / 1e6 for v in names["videos"]])
    yellow = {m: F(f"{m}_yellow") for m in L}
    red = {m: F(f"{m}_red") for m in L}

    def row(f):
        at = np.flatnonzero(video == f)
        e = {"film": f, "mbps": rnd(mbps[f], 2), "frames": int(len(at)), "size": [vf[names["videos"][f]][2], vf[names["videos"][f]][3]], "labelled": int((code[at] >= 0).sum()),
             "block": rnd(np.nanmedian(extra["block"][at]), 3), "lap": rnd(np.nanmedian(extra["lap"][at]), 1), "noise": rnd(np.nanmedian(extra["noise"][at]), 3), "spearman_u1_u3": rnd(spearmanr(L["u1"][at], L["u3"][at])[0])}
        for m in L:
            k = L[m][at] < th[m]["r98"]
            k95 = L[m][at] < th[m]["r95"]
            k90 = L[m][at] < th[m]["r90"]
            lab = code[at] >= 0
            ok = np.isfinite(region[at])
            e[m] = {"sharp_r90_r95_r98": [pct(k90.mean()), pct(k95.mean()), pct(k.mean())], "logit_median": rnd(np.median(L[m][at]), 2), "cell_yellow_median": pct(np.median(yellow[m][at])), "red_median": pct(np.median(red[m][at]), 2),
                    "logit_vs_motion": rnd(spearmanr(L[m][at][ok], region[at][ok])[0]),
                    "sss_kept_r98": pct(k[lab & (code[at] == 0)].mean()) if (lab & (code[at] == 0)).any() else None, "blurred_codes_among_kept_r98": pct(np.isin(code[at][lab & k], (3, 5)).mean()) if (lab & k).any() else None}
        return e
    high = np.flatnonzero(mbps > LIMIT)
    res = {"limit_mbps": LIMIT, "films_above": int(len(high)), "films_below": int((mbps <= LIMIT).sum()), "above": sorted((row(int(f)) for f in high), key=lambda r: -r["mbps"])}
    shift = {p: np.array([(L["u3"][video == f] < th["u3"][p]).mean() - (L["u1"][video == f] < th["u1"][p]).mean() for f in range(films)]) for p in ("r90", "r95", "r98")}
    low = mbps <= LIMIT
    res["shift_median_below"] = {p: pct(np.median(shift[p][low]), 2) for p in shift}
    res["shift_median_above"] = {p: pct(np.median(shift[p][~low]), 2) for p in shift}
    res["shift_below_p5_p95"] = {p: [pct(q, 2) for q in np.percentile(shift[p][low], [5, 95])] for p in shift}
    res["spearman_shift_mbps_all"] = {p: rnd(spearmanr(mbps, shift[p])[0]) for p in shift}
    res["spearman_shift_mbps_below"] = {p: rnd(spearmanr(mbps[low], shift[p][low])[0]) for p in shift}
    (X.DATA / "corpus_u1_u3_bitrate.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
