"""Protocol item 6, second part: the px against the measured motion inside each VLM code (held-out films).

The order of the codes in px is partly met by construction (the label trains the score the px reads); what the label
cannot force is the order inside a code. Per code, the Spearman of the measured region motion (region_px, the v9
database's camera-compensated subject motion of the labelled region) with:
- the Update 3 frame px (monotone in its logit) and the inner cells' p90 px;
- the round3 Small frame logit, as control, and its cell readouts (m_inner, max_inner; R4S-1), as the fair partner of
  the U3 cell p90; the U3 m_inner too.
Cluster bootstrap over shots (2000, within each film) for U3 minus Small.

usage: python experiments/u3r3_eval/item6.py
Output: data/item6.json
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

import common as C
import eval_v9 as E
import r4s1_small_base as S
from heldout import SCORES, SMALL_SCORES, _read


def films():
    con = sqlite3.connect(f"file:{E.V9_DB}?mode=ro", uri=True)
    region = {fid: r for fid, r in con.execute("SELECT frame_id, region_px FROM frames WHERE split = 'heldout'")}
    master = Path(json.loads(E.V9_DB.with_suffix(".json").read_text(encoding="utf-8"))["master_root"])
    db = sqlite3.connect(f"file:{master / 'master.db'}?mode=ro", uri=True)
    held = [vid for (vid,) in con.execute("SELECT DISTINCT video_id FROM frames WHERE split = 'heldout' ORDER BY video_id")]
    u3, sm = _read(SCORES), _read(SMALL_SCORES, "small_")
    out = []
    for vid in held:
        codes = E.used_codes(db, vid)
        rows = db.execute(f"SELECT id, blur_label, scene_id FROM frames WHERE video_id = ? AND "
                          f"blur_label IN ({','.join('?' * len(codes))})", (vid, *sorted(codes))).fetchall()
        f = {"code": np.array([c for _, c, _ in rows]),
             "shots": np.unique([s for *_, s in rows], return_inverse=True)[1],
             "region": np.array([np.nan if region[i] is None else region[i] for i, *_ in rows], float),
             "u3_L": np.array([u3[i]["L"] for i, *_ in rows]),
             "u3_Lc_p90": np.array([u3[i]["Lc_p90"] for i, *_ in rows]),
             "u3_m_inner": np.array([u3[i]["m_inner"] for i, *_ in rows]),
             "small_logit": np.array([sm[i]["logit"] for i, *_ in rows]),
             "small_m_inner": np.array([sm[i]["m_inner"] for i, *_ in rows]),
             "small_max_inner": np.array([sm[i]["max_inner"] for i, *_ in rows])}
        out.append(f)
    return out


def main():
    F = films()
    rng = np.random.default_rng(C.SEED)
    reads = ("u3_L", "u3_Lc_p90", "u3_m_inner", "small_logit", "small_m_inner", "small_max_inner")
    res = {"what": __doc__.split("\n\n")[0], "codes": {}}
    for code in ("sss", "ssb", "bsb", "bbb"):
        sel = [(f, np.flatnonzero((f["code"] == code) & np.isfinite(f["region"]))) for f in F]
        sel = [(f, ix) for f, ix in sel if len(ix)]
        reg = np.concatenate([f["region"][ix] for f, ix in sel])
        if len(reg) < 100:
            continue
        e = {"n": int(len(reg)), "region_px_median": float(np.median(reg))}
        for r in reads:
            e[r] = S._spearman(np.concatenate([f[r][ix] for f, ix in sel]), reg)
        groups = [[ix[f["shots"][ix] == s] for s in np.unique(f["shots"][ix])] for f, ix in sel]
        boot = {r: [] for r in reads}
        for _ in range(2000):
            parts = [(f, np.concatenate([g[j] for j in rng.integers(len(g), size=len(g))])) for (f, _), g in zip(sel, groups)]
            rg = np.concatenate([f["region"][ix] for f, ix in parts])
            for r in reads:
                boot[r].append(S._spearman(np.concatenate([f[r][ix] for f, ix in parts]), rg))
        for r in reads:
            e[f"{r}_ci95"] = np.percentile(boot[r], [2.5, 97.5]).tolist()
        for r, ref in (("u3_L", "small_logit"), ("u3_Lc_p90", "small_logit"), ("u3_Lc_p90", "small_m_inner"),
                       ("u3_m_inner", "small_m_inner"), ("u3_Lc_p90", "small_max_inner")):
            d = np.array(boot[r]) - np.array(boot[ref])
            e[f"{r}_minus_{ref}"] = {"value": e[r] - e[ref], "ci95": np.percentile(d, [2.5, 97.5]).tolist()}
        res["codes"][code] = e
        C.log(f"{code}: n {e['n']}, mozgás medián {e['region_px_median']:.2f} px | Spearman a mozgással: U3 px {e['u3_L']:.3f} "
              f"{np.round(e['u3_L_ci95'], 3).tolist()}, U3 cella-p90 {e['u3_Lc_p90']:.3f}, Small {e['small_logit']:.3f} "
              f"{np.round(e['small_logit_ci95'], 3).tolist()}, Small m_inner {e['small_m_inner']:.3f}, max_inner {e['small_max_inner']:.3f}, "
              f"U3 m_inner {e['u3_m_inner']:.3f} | " + "; ".join(
                  f"{k} {v['value']:+.3f} {np.round(v['ci95'], 3).tolist()}" for k, v in e.items() if "_minus_" in k))
    (C.DATA / "item6.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
