"""Protocol item 2 (held-out films, the R4S-1 procedure) and item 6 (px on native frames).

Every VLM-labelled frame of the six held-out films (114,687, tools/r4s1_small_base.heldout_rows) is scored with the
exported Update 3 file through the published Update 1 loader, exactly as R4S-1 scored the round3 Small; the Small's
scores of the same frames are read from R4S-1 (runs/round3_convnext_small_imagenet_p99_v8_v1/probes/r4s1_heldout.csv).
Paired differences use the same cluster-bootstrap resamples over shots (2000, within each film).

usage: python experiments/u3r3_eval/heldout.py score [--threads 6]     (resumable)
       python experiments/u3r3_eval/heldout.py report
Output: data/heldout_u3.csv, data/heldout.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import common as C
import eval_v9 as E
import r4s1_small_base as S

SCORES = C.DATA / "heldout_u3.csv"
SMALL_SCORES = S.HELDOUT
COLS = ["i", "frame_id", "logit", "m_inner", "max_inner", "mean_inner", "L", "Lc_med", "Lc_p90", "Lc_max", "Lc_ge1"]


def score(a):
    rows, master = S.heldout_rows()
    C.DATA.mkdir(parents=True, exist_ok=True)
    done = set()
    if SCORES.exists():
        with open(SCORES, newline="", encoding="utf-8") as fh:
            done = {int(r["i"]) for r in csv.DictReader(fh)}
    todo = [r for r in rows if r[0] not in done]
    C.log(f"held-out: {len(rows)} frame, kész {len(done)}, hátra {len(todo)}; Update 3 ep029 (Update 1 loader), cuda")
    model, meta = C.load_u3()
    T = torch.tensor(meta["ordinal"]["T"], device="cuda")
    b = torch.tensor(meta["ordinal"]["b"], device="cuda")
    dT = torch.diff(T, prepend=T.new_zeros(1))
    load = lambda p: np.asarray(Image.open(master / p).convert("RGB"))            # noqa: E731
    new = not SCORES.exists()
    t0 = time.time()
    with open(SCORES, "a", newline="", encoding="utf-8") as fh, ThreadPoolExecutor(a.threads) as pool:
        w = csv.DictWriter(fh, COLS)
        if new:
            w.writeheader()
        ahead = [pool.submit(load, r[2]) for r in todo[:2 * a.threads]]
        for n, (i, fid, *_) in enumerate(todo):
            rgb = ahead.pop(0).result()
            if n + 2 * a.threads < len(todo):
                ahead.append(pool.submit(load, todo[n + 2 * a.threads][2]))
            x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].cuda()
            out = C.forward(model, x)
            st = S.stats(out)
            inner = out["local"].float()[0, 0, 1:-1, 1:-1]
            Lc = (torch.sigmoid(inner.unsqueeze(-1) - b) * dT).sum(-1).flatten()
            q = torch.quantile(Lc, torch.tensor([0.5, 0.9], device="cuda"))
            row = {"i": i, "frame_id": fid, **{k: f"{v:.7g}" for k, v in st.items()},
                   "L": f"{float(C.measured_L(st['logit'], meta)):.5g}", "Lc_med": f"{float(q[0]):.5g}",
                   "Lc_p90": f"{float(q[1]):.5g}", "Lc_max": f"{float(Lc.max()):.5g}",
                   "Lc_ge1": f"{float((Lc >= 1).float().mean()):.5g}"}
            w.writerow(row)
            if (n + 1) % 1000 == 0:
                fh.flush()
                left = (time.time() - t0) / (n + 1) * (len(todo) - n - 1)
                C.log(f"  {n + 1} / {len(todo)}, hátra kb. {left / 60:.0f} perc")
    C.log(f"kész: {SCORES}")


def _read(path, prefix=""):
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["frame_id"]] = {k[len(prefix):]: float(v) for k, v in r.items()
                                  if k not in ("i", "frame_id") and k.startswith(prefix)}
    return out


def report(a):
    published = json.loads((C.UP / "frame-blur-detection-update-1" / "reproduction" / "heldout_films.json").read_text(encoding="utf-8"))
    rules = json.loads(Path("F:/RGB Mesh/dataset/frames/selection_rules_v8.json").read_text(encoding="utf-8"))
    con = sqlite3.connect(f"file:{E.V9_DB}?mode=ro", uri=True)
    region = {fid: (r, ev) for fid, r, ev in con.execute("SELECT frame_id, region_px, eval FROM frames WHERE split = 'heldout'")}
    master = Path(json.loads(E.V9_DB.with_suffix(".json").read_text(encoding="utf-8"))["master_root"])
    db = sqlite3.connect(f"file:{master / 'master.db'}?mode=ro", uri=True)
    names = {vid: Path(src).name for vid, src in db.execute("SELECT id, source_path FROM videos")}
    held = [vid for (vid,) in con.execute("SELECT DISTINCT video_id FROM frames WHERE split = 'heldout' ORDER BY video_id")]
    role = {vid: "test" if names[vid] in set(rules["split"]["test"]) else "validation" for vid in held}
    u3, sm = _read(SCORES), _read(SMALL_SCORES, "small_")
    _, mu = C.load_u3("cpu")
    _, ms = C.load_small("cpu")
    metas = {"u3": mu, "small": ms}
    films = {}
    for vid in held:
        codes = E.used_codes(db, vid)
        rows = db.execute(f"SELECT id, blur_label, scene_id FROM frames WHERE video_id = ? AND "
                          f"blur_label IN ({','.join('?' * len(codes))})", (vid, *sorted(codes))).fetchall()
        miss = sum(1 for r in rows if r[0] not in u3 or r[0] not in sm)
        if miss:
            raise SystemExit(f"{miss} labelled frames of a held-out film are not scored yet")
        f = {"blurred": np.array([c in ("bbb", "bsb") for _, c, _ in rows], bool),
             "code": np.array([c for _, c, _ in rows]),
             "shots": np.unique([s for *_, s in rows], return_inverse=True)[1],
             "region": np.array([np.nan if region[i][0] is None else region[i][0] for i, *_ in rows], float),
             "eval": np.array([bool(region[i][1]) for i, *_ in rows])}
        for k in ("logit", "m_inner", "L", "Lc_med", "Lc_p90", "Lc_max", "Lc_ge1"):
            f[f"u3_{k}"] = np.array([u3[i][k] for i, *_ in rows])
        for k in ("logit", "m_inner"):
            f[f"small_{k}"] = np.array([sm[i][k] for i, *_ in rows])
        films[vid] = f
    by_counts = {(p["n_sharp"], p["n_blur"]): p["film"] for p in published["films"]}
    label = {vid: by_counts[(int((~f["blurred"]).sum()), int(f["blurred"].sum()))] for vid, f in films.items()}
    rng = np.random.default_rng(C.SEED)
    readouts = {"u3": "u3_logit", "small": "small_logit"}

    def group(members):
        keys = [k for k in films[members[0]]]
        cat = {k: np.concatenate([films[v][k] for v in members]) for k in keys if k != "shots"}
        b = cat["blurred"]
        out = {"films": [label[v] for v in members], "n_sharp": int((~b).sum()), "n_blur": int(b.sum()),
               "auc": {m: E._auc(cat[c], b) for m, c in readouts.items()}}
        groups = [[np.flatnonzero(films[v]["shots"] == s) for s in range(films[v]["shots"].max() + 1)] for v in members]
        boot = {m: [] for m in readouts}
        rho = {m: [] for m in readouts}
        kept = {(m, p): [] for m in readouts for p in ("r90", "r95", "r98")}
        for _ in range(a.reps):
            parts = []
            for v, g in zip(members, groups):
                pick = rng.integers(len(g), size=len(g))
                parts.append((v, np.concatenate([g[j] for j in pick])))
            bb = np.concatenate([films[v]["blurred"][ix] for v, ix in parts])
            reg = np.concatenate([films[v]["region"][ix] for v, ix in parts])
            ok = np.isfinite(reg)
            for m, c in readouts.items():
                s = np.concatenate([films[v][c][ix] for v, ix in parts])
                if 0 < bb.sum() < len(bb):
                    boot[m].append(E._auc(s, bb))
                rho[m].append(S._spearman(s[ok], reg[ok]))
                for p in ("r90", "r95", "r98"):
                    kept[(m, p)].append(float(np.mean(s[~bb] < C.logit_threshold(metas[m], p))))
        out["auc_ci95"] = {m: np.percentile(v, [2.5, 97.5]).tolist() for m, v in boot.items()}
        d = np.array(boot["u3"]) - np.array(boot["small"])
        out["auc_difference_u3_minus_small"] = {"value": out["auc"]["u3"] - out["auc"]["small"],
                                                "ci95": np.percentile(d, [2.5, 97.5]).tolist()}
        out["operating_points"] = {}
        for m in readouts:
            for p in ("r90", "r95", "r98"):
                call = cat[readouts[m]] >= C.logit_threshold(metas[m], p)
                out["operating_points"][f"{m}_{p}"] = {"threshold": metas[m]["thresholds"][p],
                                                       "sharp_kept": float(np.mean(~call[~b])),
                                                       "sharp_kept_ci95": np.percentile(kept[(m, p)], [2.5, 97.5]).tolist(),
                                                       "blur_caught": float(np.mean(call[b]))}
        for p in ("r90", "r95", "r98"):
            dk = np.array(kept[("u3", p)]) - np.array(kept[("small", p)])
            out["operating_points"][f"sharp_kept_difference_u3_minus_small_{p}"] = {
                "value": out["operating_points"][f"u3_{p}"]["sharp_kept"] - out["operating_points"][f"small_{p}"]["sharp_kept"],
                "ci95": np.percentile(dk, [2.5, 97.5]).tolist(),
                "note": "the blurred caught differs between the files (each has its own validation thresholds)"}
        ok = np.isfinite(cat["region"])
        dr = np.array(rho["u3"]) - np.array(rho["small"])
        out["motion"] = {"frames_with_measurement": int(ok.sum()),
                         "spearman_logit_region_px": {m: S._spearman(cat[c][ok], cat["region"][ok]) for m, c in readouts.items()},
                         "spearman_ci95": {m: np.percentile(v, [2.5, 97.5]).tolist() for m, v in rho.items()},
                         "spearman_difference_u3_minus_small": {"ci95": np.percentile(dr, [2.5, 97.5]).tolist()},
                         "spearman_m_inner_region_px": {m: S._spearman(cat[f"{m}_m_inner"][ok], cat["region"][ok]) for m in readouts},
                         "spearman_cell_px_p90_region_px_u3": S._spearman(cat["u3_Lc_p90"][ok], cat["region"][ok])}
        cu, cs = cat["u3_logit"] >= C.logit_threshold(mu, "r95"), cat["small_logit"] >= C.logit_threshold(ms, "r95")
        dis = cu != cs
        out["agreement_r95"] = {"same_decision": float(np.mean(~dis)), "cohen_kappa": S._kappa(cu, cs),
                                "spearman_logits": S._spearman(cat["u3_logit"], cat["small_logit"]),
                                "disagreements": int(dis.sum()),
                                "of_which_the_vlm_label_sides_with_u3": float(np.mean(cu[dis] == b[dis])) if dis.any() else None}
        # item 6: px by VLM code (frame level and cells)
        px = {}
        for code in ("sss", "ssb", "bsb", "bbb"):
            s = cat["code"] == code
            if s.sum() < 20:
                continue
            px[code] = {"n": int(s.sum()), "L_median": float(np.median(cat["u3_L"][s])),
                        "L_p10_p90": np.quantile(cat["u3_L"][s], [0.1, 0.9]).tolist(),
                        "L_under_1px": float(np.mean(cat["u3_L"][s] < 1)),
                        "cell_p90_median": float(np.median(cat["u3_Lc_p90"][s])),
                        "cell_max_median": float(np.median(cat["u3_Lc_max"][s])),
                        "cell_max_under_1px": float(np.mean(cat["u3_Lc_max"][s] < 1)),
                        "cells_ge_1px_share_median": float(np.median(cat["u3_Lc_ge1"][s]))}
        order = {}
        for hi, lo in (("bbb", "bsb"), ("bsb", "sss"), ("bbb", "sss")):
            if hi in px and lo in px:
                order[f"{hi}>{lo}"] = {k: E._auc(np.concatenate([cat[k][cat["code"] == lo], cat[k][cat["code"] == hi]]),
                                                 np.r_[np.zeros((cat["code"] == lo).sum(), bool), np.ones((cat["code"] == hi).sum(), bool)])
                                       for k in ("u3_L", "u3_Lc_p90", "u3_Lc_max")}
        out["px_by_code"] = px
        out["px_order_auc"] = order
        return out

    val = [v for v in held if role[v] == "validation"]
    test = [v for v in held if role[v] == "test"]
    G = {"validation films (3)": group(val), "test films (3)": group(test), "validation + test (6 films)": group(held)}
    res = {"what": "Protocol items 2 and 6: the six held-out films, Update 3 ep029 vs the round3 Small ep029 (R4S-1 scores), "
                   "both through the published Update 1 loader; VLM labels as in Update 1 (blurred = bbb + bsb)",
           "bootstrap": f"{a.reps} cluster resamples over shots within each film, seed {C.SEED}, the same for both models",
           "groups": [{"group": k, **v} for k, v in G.items()],
           "films": [{"film": label[v], "role": role[v], "n_sharp": int((~films[v]["blurred"]).sum()),
                      "n_blur": int(films[v]["blurred"].sum()),
                      "auc": {m: E._auc(films[v][c], films[v]["blurred"]) for m, c in readouts.items()}} for v in held]}
    (C.DATA / "heldout.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    for g in res["groups"]:
        d = g["auc_difference_u3_minus_small"]
        C.log(f"{g['group']}: AUC U3 {g['auc']['u3']:.4f} Small {g['auc']['small']:.4f}, különbség {d['value']:+.4f} "
              f"[{d['ci95'][0]:+.4f}; {d['ci95'][1]:+.4f}]")
        C.log("   éles megtartva r90/r95/r98: U3 " + " / ".join(f"{g['operating_points'][f'u3_{p}']['sharp_kept']:.3f}" for p in ("r90", "r95", "r98"))
              + ", Small " + " / ".join(f"{g['operating_points'][f'small_{p}']['sharp_kept']:.3f}" for p in ("r90", "r95", "r98")))
        C.log(f"   mozgás Spearman: {json.dumps(g['motion']['spearman_logit_region_px'])}; px kódonként: "
              + "; ".join(f"{c} {v['L_median']:.2f} px (<1 px {v['L_under_1px']:.1%})" for c, v in g["px_by_code"].items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("score", "report"))
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()
    {"score": score, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
