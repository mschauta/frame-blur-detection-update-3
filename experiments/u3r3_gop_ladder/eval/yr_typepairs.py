"""Protocol item 5, added after the first episode was scored (napló 2026-10-30): the same frame, a different frame type.

On film an I-frame can be sharper for real: the source encode gives a keyframe more bits, and the frames after it
lose detail step by step. The pre-registered measures (agreement, Phase II enrichment) cannot tell this from a model
bias, because the keyframes of the codecs mostly sit at the same frames. A frame that is an I-frame in one codec and a
P- or B-frame in the other can: the picture is the same, only the coding differs.

Per episode and codec pair (aligned frames, both lit), grouped by the pair of frame types:
- the median logit difference (codec x minus codec y) of each model, and the share where x reads sharper;
- the double difference against the frames that are non-I in both codecs (the codec's own offset removed);
- the U3 px of both.

usage: python experiments/u3r3_eval/yr_typepairs.py
Output: data/yr_typepairs.json
"""
from __future__ import annotations

import json

import numpy as np

import common as C
from yr import VIDEOS


def main():
    D = {lab: dict(np.load(C.DATA / f"yr_{lab}.npz", allow_pickle=False)) for lab in VIDEOS if (C.DATA / f"yr_{lab}.npz").exists()}
    rng = np.random.default_rng(C.SEED)
    res = {}
    for ep in sorted({lab.split("_")[0] for lab in D}):
        labs = [lab for lab in D if lab.startswith(ep)]
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                x, y = D[labs[i]], D[labs[j]]
                if len(x["type"]) != len(y["type"]):
                    continue
                lit = (x["luma"] >= 8) & (y["luma"] >= 8)
                tx, ty = x["type"], y["type"]
                both_non_i = lit & (tx != "I") & (ty != "I")
                out = {}
                for gx in "IPB":
                    for gy in "IPB":
                        sel = lit & (tx == gx) & (ty == gy)
                        if sel.sum() < 20:
                            continue
                        e = {"n": int(sel.sum())}
                        for m in ("u3", "small"):
                            d = x[f"{m}_logit"] - y[f"{m}_logit"]
                            base = float(np.median(d[both_non_i]))
                            idx = np.flatnonzero(sel)
                            boot = [np.median(d[rng.choice(idx, len(idx))]) for _ in range(1000)]
                            e[m] = {"median_logit_x_minus_y": float(np.median(d[sel])),
                                    "ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
                                    "x_sharper_share": float(np.mean(d[sel] < 0)),
                                    "minus_non_i_baseline": float(np.median(d[sel]) - base)}
                        e["u3_px_x_median"] = float(np.median(x["u3_L"][sel]))
                        e["u3_px_y_median"] = float(np.median(y["u3_L"][sel]))
                        e["pkt_x_median"] = int(np.median(x["pkt_size"][sel]))
                        e["pkt_y_median"] = int(np.median(y["pkt_size"][sel]))
                        out[f"{gx}~{gy}"] = e
                res[f"{labs[i]}~{labs[j]}"] = out
    (C.DATA / "yr_typepairs.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, out in res.items():
        C.log(k)
        for g, e in out.items():
            C.log(f"  {g:4s} n {e['n']:6d}: " + "; ".join(
                f"{m} Δlogit {e[m]['median_logit_x_minus_y']:+6.2f} {np.round(e[m]['ci95'], 2).tolist()} (x élesebb {e[m]['x_sharper_share']:.0%}, "
                f"alapvonal nélkül {e[m]['minus_non_i_baseline']:+6.2f})" for m in ("u3", "small"))
                + f" | U3 px {e['u3_px_x_median']:.2f} / {e['u3_px_y_median']:.2f}, csomag {e['pkt_x_median']} / {e['pkt_y_median']}")


if __name__ == "__main__":
    main()
