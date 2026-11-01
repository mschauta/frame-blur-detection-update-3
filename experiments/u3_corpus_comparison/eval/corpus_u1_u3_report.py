"""The tables of the corpus comparison (corpus_u1_u3.py, napló 2026-11-01 23:35), from the extracted cache.

Models: u1 = Update 1 Base v1.0, u3 = Update 3 Small. Sharp = logit below the model's own threshold (r90 / r95 / r98).
"eq" = per film, u3 keeps exactly as many frames as u1 keeps at that point. Differences are u3 - u1, the interval is a
bootstrap over films (2000, seed 20261030). Writes data/corpus_u1_u3.json; film names are not written.
"""
from __future__ import annotations

import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

import corpus_u1_u3 as X

POINTS = ("r90", "r95", "r98")
CODES = ("sss", "ssb", "bss", "bsb", "sbb", "bbb")
MIN_SHOT = 8
EXTRA = {"noise": "noise_estimate", "block": "compression_blockiness", "lum": "luminance_mean", "lstd": "luminance_std",
         "lap": "laplacian_variance", "entropy": "entropy_bits", "over": "overexposed_fraction", "under": "underexposed_fraction",
         "edge": "edge_density"}


def pct(x, d=1):
    return round(float(x) * 100, d)


def rnd(x, d=3):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def boot(values, seed=X.SEED, n=2000):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    meds = np.median(v[rng.integers(len(v), size=(n, len(v)))], axis=1)
    return {"median": rnd(np.median(v), 4), "ci95": [rnd(q, 4) for q in np.percentile(meds, [2.5, 97.5])], "films": int(len(v)), "positive": int((v > 0).sum())}


def kappa(a, b):
    agree = np.mean(a == b)
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float(agree), float((agree - pe) / (1 - pe)) if pe < 1 else float("nan")


def two_by_two(a, b):
    agree, k = kappa(a, b)
    return {"both_sharp": pct((a & b).mean()), "only_u1_sharp": pct((a & ~b).mean()), "only_u3_sharp": pct((~a & b).mean()), "both_blurred": pct((~a & ~b).mean()),
            "agree": pct(agree), "kappa": rnd(k)}


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return (a * b).sum() / d if d > 0 else np.nan


def extra_columns(a, ids):
    """Frame facts beyond the extraction (image statistics that may explain a difference), cached beside it."""
    path = Path(a.cache) / "extra.npz"
    if path.exists():
        return dict(np.load(path))
    X.log("extra columns")
    con = X.connect()
    sel = ",".join(f"json_extract(quality_metrics,'$.{v}')" for v in EXTRA.values())
    out = {k: np.full(len(ids), np.nan, np.float32) for k in EXTRA}
    cur = con.execute(f"select id,{sel} from frames")
    while True:
        part = cur.fetchmany(500_000)
        if not part:
            break
        at = np.searchsorted(ids, np.array([p[0] for p in part], "S36"))
        for j, k in enumerate(EXTRA):
            out[k][at] = np.array([np.nan if p[j + 1] is None else p[j + 1] for p in part], np.float32)
    np.savez(path, **out)
    return out


def report(a):
    cache = Path(a.cache)
    ids = np.load(cache / "ids.npy")
    facts = dict(np.load(cache / "facts.npz"))
    names = json.loads((cache / "names.json").read_text(encoding="utf-8"))
    feats = np.load(cache / "features.npy", mmap_mode="r")
    col = {n: i for i, n in enumerate(json.loads((cache / "features.json").read_text(encoding="utf-8")))}
    F = lambda n: np.asarray(feats[:, col[n]])       # noqa: E731
    extra = extra_columns(a, ids)
    N = len(ids)
    L = {"u1": facts["logit_u1"], "u3": facts["logit_u3"]}
    video, scene, fidx = facts["video"], facts["scene"], facts["fidx"]
    films = int(video.max()) + 1
    th = {m: {p: float(np.log(t / (1 - t))) for p, t in names["thresholds"][m].items()} for m in X.SETS}
    res = {"frames": N, "films": films, "thresholds_logit": {m: {p: rnd(v) for p, v in th[m].items()} for m in th}}
    X.log(f"frames {N:,}, films {films}")

    # ---- groups: the round3 index both models learned from
    split = np.zeros(N, np.int8)           # 0 not in the index, 1 train, 2 val, 3 test
    target = np.full(N, -1, np.int8)
    val_target = {}
    with open(X.INDEX, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["source"] == "frames"]
    at = np.searchsorted(ids, np.array([r["sample_id"].split("/", 1)[1] for r in rows], "S36"))
    assert (ids[at] == np.array([r["sample_id"].split("/", 1)[1] for r in rows], "S36")).all()
    split[at] = [{"train": 1, "val": 2, "test": 3}[r["split"]] for r in rows]
    target[at] = [int(r["target"]) for r in rows]
    held_films = np.unique(video[split >= 2])
    assert not np.isin(video[split == 1], held_films).any(), "a held-out film has training frames"
    held = np.isin(video, held_films)
    groups = {"all": np.ones(N, bool), "heldout": held, "trained": split == 1, "rest": ~held & (split != 1)}
    res["groups"] = {k: int(v.sum()) for k, v in groups.items()}
    res["heldout_films"] = int(len(held_films))
    X.log(f"groups {res['groups']}, held-out films {len(held_films)}")

    # the stored scores must reproduce each run's validation AUC
    def auc(score, y):
        r = rankdata(score)
        pos = y == 1
        return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))
    res["auc"] = {}
    for sp, name in ((2, "val"), (3, "test")):
        m = split == sp
        res["auc"][name] = {k: rnd(auc(L[k][m], target[m]), 5) for k in L}
    con = X.connect()
    res["auc"]["val_in_run_meta"] = {m: rnd(json.loads(con.execute("select meta from blur_model_sets where set_id=?", (s,)).fetchone()[0]).get("val_frames_auc"), 5) for m, s in X.SETS.items()}
    X.log(f"AUC {res['auc']}")

    # ---- shot facts: the close-up proxy, shot rows
    order = np.lexsort((fidx, scene))
    sc_sorted = scene[order]
    starts = np.flatnonzero(np.r_[True, sc_sorted[1:] != sc_sorted[:-1]])
    ends = np.r_[starts[1:], N]
    kept = facts["selected"] & ~facts["teacher_branch"]
    shot_kept = np.bincount(scene[scene >= 0], kept[scene >= 0], minlength=int(scene.max()) + 1)
    zero_shot = (scene >= 0) & (shot_kept[np.maximum(scene, 0)] == 0)
    res["zero_shot_share"] = pct(zero_shot.mean())
    res["no_scene_frames"] = int((scene < 0).sum())
    gaps = np.diff(fidx[order])[sc_sorted[1:] == sc_sorted[:-1]]
    res["adjacent_gap"] = {"median": int(np.median(gaps)), "share_gap_1": pct((gaps == 1).mean()), "p99": int(np.percentile(gaps, 99))}
    X.log(f"zero shots {res['zero_shot_share']}%, gaps {res['adjacent_gap']}")

    # ---- decisions
    keep = {m: {p: L[m] < th[m][p] for p in POINTS} for m in L}
    film_at = [np.flatnonzero(video == f) for f in range(films)]
    eq = {}
    for p in POINTS:
        k = np.zeros(N, bool)
        for at_f in film_at:
            n = int(keep["u1"][p][at_f].sum())
            if n:
                k[at_f[np.argpartition(L["u3"][at_f], n - 1)[:n]]] = True
        eq[p] = k
    assert all(eq[p].sum() == keep["u1"][p].sum() for p in POINTS)
    X.log("decisions")

    def per_film(values, weight=None):
        c = np.bincount(video, values.astype(np.float64) if weight is None else values.astype(np.float64) * weight, minlength=films)
        n = np.bincount(video, None if weight is None else weight.astype(np.float64), minlength=films)
        with np.errstate(invalid="ignore", divide="ignore"):
            return c / n

    # 1. threshold
    res["sharp_share"] = {}
    for g, m in groups.items():
        res["sharp_share"][g] = {p: {k: pct(keep[k][p][m].mean()) for k in L} for p in POINTS}
    res["sharp_share_film"] = {}
    for p in POINTS:
        d = per_film(keep["u3"][p]) - per_film(keep["u1"][p])
        res["sharp_share_film"][p] = {"u1_p5_p50_p95": [pct(q) for q in np.percentile(per_film(keep["u1"][p]), [5, 50, 95])],
                                      "u3_p5_p50_p95": [pct(q) for q in np.percentile(per_film(keep["u3"][p]), [5, 50, 95])],
                                      "diff": boot(d), "diff_min_max": [pct(d.min()), pct(d.max())], "diff_abs_median": pct(np.median(np.abs(d)))}
    X.log(f"sharp share {res['sharp_share']['all']}")

    # 2. confidence
    res["logit_quantiles"] = {}
    qs = [1, 5, 25, 50, 75, 95, 99]
    for g, m in groups.items():
        res["logit_quantiles"][g] = {k: [rnd(q, 2) for q in np.percentile(L[k][m], qs)] for k in L}
    res["band_share"] = {g: {k: pct(((L[k][m] >= th[k]["r98"]) & (L[k][m] < th[k]["r90"])).mean()) for k in L} for g, m in groups.items()}
    bd = per_film((L["u3"] >= th["u3"]["r98"]) & (L["u3"] < th["u3"]["r90"])) - per_film((L["u1"] >= th["u1"]["r98"]) & (L["u1"] < th["u1"]["r90"]))
    res["band_share_diff"] = boot(bd)
    X.log(f"band {res['band_share']}")

    # 3. agreement
    res["spearman_global"] = {g: rnd(spearmanr(L["u1"][m], L["u3"][m])[0], 4) for g, m in groups.items()}
    film_sp = np.array([spearmanr(L["u1"][i], L["u3"][i])[0] for i in film_at])
    res["spearman_film"] = {"median": rnd(np.median(film_sp), 4), "p5_p95": [rnd(q, 4) for q in np.percentile(film_sp, [5, 95])], "min": rnd(film_sp.min(), 4)}
    res["two_by_two_own"] = {g: {p: two_by_two(keep["u1"][p][m], keep["u3"][p][m]) for p in POINTS} for g, m in groups.items()}
    res["two_by_two_eq"] = {g: {p: two_by_two(keep["u1"][p][m], eq[p][m]) for p in POINTS} for g, m in groups.items()}
    X.log(f"spearman {res['spearman_global']}, film {res['spearman_film']}")

    # shot loop: within-shot Spearman, motion rank
    motion = {"region": facts["region"], "subject": facts["subject"], "camera": facts["camera"]}
    shot_stats = defaultdict(list)
    shot_film, shot_size, shot_zero = [], [], []
    prank = np.full(N, np.nan, np.float32)        # percent rank of region motion inside the shot
    r1 = np.full(N, np.nan, np.float32)           # logit ranks inside the shot, in the same percent scale
    r3 = np.full(N, np.nan, np.float32)
    for s, e in zip(starts, ends):
        at_s = order[s:e]
        if scene[at_s[0]] < 0 or e - s < MIN_SHOT:
            continue
        a1, a3 = rankdata(L["u1"][at_s]), rankdata(L["u3"][at_s])
        r1[at_s], r3[at_s] = a1 / (e - s), a3 / (e - s)
        shot_film.append(video[at_s[0]])
        shot_size.append(e - s)
        shot_zero.append(zero_shot[at_s[0]])
        shot_stats["u1_u3"].append(corr(a1, a3))
        for name, v in motion.items():
            x = v[at_s]
            ok = np.isfinite(x)
            if ok.sum() >= MIN_SHOT:
                rx = rankdata(x[ok])
                if name == "region":
                    prank[at_s[ok]] = rx / ok.sum()
                shot_stats[f"u1_{name}"].append(corr(rankdata(a1[ok]), rx))
                shot_stats[f"u3_{name}"].append(corr(rankdata(a3[ok]), rx))
            else:
                shot_stats[f"u1_{name}"].append(np.nan)
                shot_stats[f"u3_{name}"].append(np.nan)
    shot_film, shot_zero = np.array(shot_film), np.array(shot_zero)
    shot_stats = {k: np.array(v, float) for k, v in shot_stats.items()}
    held_shot = np.isin(shot_film, held_films)
    res["shots_used"] = {"shots": int(len(shot_film)), "min_frames": MIN_SHOT, "share_zero": pct(shot_zero.mean())}
    res["within_shot"] = {"u1_u3": {"all": rnd(np.nanmedian(shot_stats["u1_u3"])), "heldout": rnd(np.nanmedian(shot_stats["u1_u3"][held_shot])),
                                    "zero_shots": rnd(np.nanmedian(shot_stats["u1_u3"][shot_zero])), "other_shots": rnd(np.nanmedian(shot_stats["u1_u3"][~shot_zero]))}}
    film_med = lambda arr: np.array([np.nanmedian(arr[shot_film == f]) if (shot_film == f).any() and np.isfinite(arr[shot_film == f]).any() else np.nan for f in range(films)])       # noqa: E731
    for name in motion:
        d1, d3 = film_med(shot_stats[f"u1_{name}"]), film_med(shot_stats[f"u3_{name}"])
        res["within_shot"][name] = {"u1_all": rnd(np.nanmedian(shot_stats[f"u1_{name}"])), "u3_all": rnd(np.nanmedian(shot_stats[f"u3_{name}"])),
                                    "u1_heldout": rnd(np.nanmedian(shot_stats[f"u1_{name}"][held_shot])), "u3_heldout": rnd(np.nanmedian(shot_stats[f"u3_{name}"][held_shot])),
                                    "u1_zero": rnd(np.nanmedian(shot_stats[f"u1_{name}"][shot_zero])), "u3_zero": rnd(np.nanmedian(shot_stats[f"u3_{name}"][shot_zero])),
                                    "u3_minus_u1_films": boot(d3 - d1)}
    X.log(f"within shot {res['within_shot']}")

    # 4. hard contradictions
    sure = {"u1_sharp_u3_blurred": (L["u1"] < th["u1"]["r98"]) & (L["u3"] >= th["u3"]["r90"]), "u1_blurred_u3_sharp": (L["u1"] >= th["u1"]["r90"]) & (L["u3"] < th["u3"]["r98"])}
    code = facts["code"]
    res["hard"] = {}
    for name, m in sure.items():
        e = {"frames": int(m.sum()), "per_mille": rnd(m.mean() * 1000, 2), "groups_per_mille": {g: rnd(m[gm].mean() * 1000, 2) for g, gm in groups.items()},
             "zero_shot_share": pct(zero_shot[m].mean()), "labelled": int((code[m] >= 0).sum()), "code_share": {c: pct((code[m] == i).sum() / max((code[m] >= 0).sum(), 1)) for i, c in enumerate(CODES)},
             "region_median": rnd(np.nanmedian(motion["region"][m]), 2), "subject_median": rnd(np.nanmedian(motion["subject"][m]), 2), "camera_median": rnd(np.nanmedian(motion["camera"][m]), 2),
             "motion_rank_in_shot_median": rnd(np.nanmedian(prank[m]), 3)}
        fr = per_film(m)
        e["films_per_mille_p50_p95_max"] = [rnd(q * 1000, 2) for q in (np.median(fr), np.percentile(fr, 95), fr.max())]
        res["hard"][name] = e
    X.log(f"hard {res['hard']}")

    # the tiers proposed by the user (napló 2026-11-01): A below r98, B r98..r95, C r95..r90, D from r90
    tier = {m: np.digitize(L[m], [th[m]["r98"], th[m]["r95"], th[m]["r90"]]).astype(np.int8) for m in L}
    tier_groups = {**groups, "zero_shots": zero_shot, "other_shots": ~zero_shot & (scene >= 0)}
    red = {m: F(f"{m}_red") for m in L}
    peak = {m: F(f"{m}_max") for m in L}

    def tier_row(k, m):
        lab = k & (code >= 0)
        return {"frames": int(k.sum()), "labelled": int(lab.sum()), "sss": pct((code[lab] == 0).mean()) if lab.any() else None, "bss": pct((code[lab] == 2).mean()) if lab.any() else None,
                "blurred_codes": pct(np.isin(code[lab], (3, 5)).mean()) if lab.any() else None, "bbb": pct((code[lab] == 5).mean()) if lab.any() else None, "zero_shot_share": pct(zero_shot[k].mean()) if k.any() else None,
                "region_median": rnd(np.nanmedian(motion["region"][k]), 2), "subject_median": rnd(np.nanmedian(motion["subject"][k]), 2), "camera_median": rnd(np.nanmedian(motion["camera"][k]), 2),
                "red_median_pct": pct(np.nanmedian(red[m][k]), 2), "no_red_pct": pct((red[m][k] == 0).mean()), "max_median": rnd(np.nanmedian(peak[m][k]), 2)}
    res["tiers"] = {}
    for g, gm in tier_groups.items():
        res["tiers"][g] = {m: {T: {"share": pct((gm & (tier[m] == i)).sum() / gm.sum()), **tier_row(gm & (tier[m] == i), m)} for i, T in enumerate("ABCD")} for m in L}
    res["tier_cross"] = {}
    for g, gm in tier_groups.items():
        c = np.bincount(tier["u1"][gm].astype(np.int32) * 4 + tier["u3"][gm], minlength=16).reshape(4, 4).astype(np.float64)
        res["tier_cross"][g] = {"percent": (c / c.sum() * 100).round(2).tolist(), "row_percent_u3_given_u1": (c / c.sum(1, keepdims=True) * 100).round(1).tolist(), "same_tier": pct(np.trace(c) / c.sum()),
                                "within_one": pct(sum(c[i, j] for i in range(4) for j in range(4) if abs(i - j) <= 1) / c.sum()), "two_or_more_apart": pct(sum(c[i, j] for i in range(4) for j in range(4) if abs(i - j) >= 2) / c.sum())}
    res["tier_films"] = {}
    for i, T in enumerate("ABCD"):
        s1, s3 = per_film(tier["u1"] == i), per_film(tier["u3"] == i)
        res["tier_films"][T] = {"u1_p5_p50_p95": [pct(q) for q in np.percentile(s1, [5, 50, 95])], "u3_p5_p50_p95": [pct(q) for q in np.percentile(s3, [5, 50, 95])], "u3_minus_u1": boot(s3 - s1)}
    n_scenes = int(scene.max()) + 1
    has_scene = scene >= 0
    res["tier_shots"] = {}
    scene_zero = np.zeros(n_scenes, bool)
    scene_zero[scene[zero_shot]] = True
    scene_used = np.bincount(scene[has_scene], minlength=n_scenes) > 0
    for m in L:
        best = np.full(n_scenes, 3, np.int8)
        np.minimum.at(best, scene[has_scene], tier[m][has_scene])
        e = {}
        for name, sm in (("all", scene_used), ("zero_shots", scene_used & scene_zero), ("other_shots", scene_used & ~scene_zero)):
            e[name] = {"shots": int(sm.sum()), "best_tier_share": {T: pct((best[sm] == i).mean()) for i, T in enumerate("ABCD")}}
        best_of = np.full(N, 3, np.int8)
        best_of[has_scene] = best[scene[has_scene]]
        isC = tier[m] == 2
        e["C_frames"] = {}
        for name, gm in (("zero_shots", zero_shot), ("other_shots", ~zero_shot & has_scene)):
            e["C_frames"][name] = {"shot_has_better_tier": tier_row(isC & gm & (best_of < 2), m), "C_is_shot_best": tier_row(isC & gm & (best_of == 2), m)}
        res["tier_shots"][m] = e
    X.log(f"tiers all {res['tiers']['all']}")
    X.log(f"tier cross {res['tier_cross']['all']}")
    X.log(f"tier shots {res['tier_shots']}")

    # where the two differ, own r95 and eq r95
    def strata_table(mask_of):
        out = {}
        for name, m in mask_of.items():
            if m.sum() == 0:
                continue
            out[name] = {"frames": int(m.sum()), "own_r95": two_by_two(keep["u1"]["r95"][m], keep["u3"]["r95"][m]), "eq_r95": two_by_two(keep["u1"]["r95"][m], eq["r95"][m])}
        return out
    bins = lambda v, edges: {f"{a}-{b}": (v >= a) & (v < b) for a, b in zip(edges[:-1], edges[1:])}       # noqa: E731
    res["strata"] = {"zero_shot": strata_table({"zero": zero_shot, "other": ~zero_shot & (scene >= 0)}),
                     "region": strata_table(bins(motion["region"], (0, 2, 4, 8, 16, 1e9))), "camera": strata_table(bins(motion["camera"], (0, 1, 2, 4, 8, 16, 1e9))),
                     "subject": strata_table(bins(motion["subject"], (0, 1, 2, 4, 8, 16, 1e9))),
                     "eligible": strata_table({"eligible": facts["eligible"], "not": ~facts["eligible"]}), "subtitle": strata_table({"yes": facts["subtitle"], "no": ~facts["subtitle"]}),
                     "overlay": strata_table({"yes": facts["overlay"], "no": ~facts["overlay"]})}
    X.log("strata")

    # what the rank difference follows, inside a film
    pr1, pr3 = np.zeros(N, np.float32), np.zeros(N, np.float32)
    for at_f in film_at:
        pr1[at_f], pr3[at_f] = rankdata(L["u1"][at_f]) / len(at_f), rankdata(L["u3"][at_f]) / len(at_f)
    dr = pr3 - pr1
    covars = {**{k: extra[k] for k in EXTRA}, **motion, "blur_area": facts["blur_area"], "u1_red": F("u1_red"), "u3_red": F("u3_red"), "u1_max": F("u1_max"), "u3_max": F("u3_max")}
    res["rank_difference_follows"] = {}
    for name, v in covars.items():
        sp = []
        for at_f in film_at:
            ok = np.isfinite(v[at_f]) & np.isfinite(dr[at_f])
            sp.append(spearmanr(dr[at_f][ok], v[at_f][ok])[0] if ok.sum() > 100 else np.nan)
        sp = np.array(sp)
        res["rank_difference_follows"][name] = boot(sp)
    res["rank_difference_abs"] = {"median": rnd(np.median(np.abs(dr)), 4), "p90": rnd(np.percentile(np.abs(dr), 90), 4), "p99": rnd(np.percentile(np.abs(dr), 99), 4)}
    X.log(f"rank difference {res['rank_difference_abs']}: { {k: v['median'] for k, v in res['rank_difference_follows'].items()} }")

    # films: how the sharp-share shift follows the film's bitrate, resolution
    vid_facts = {v: r for v, *r in con.execute("select id, source_path, source_size_bytes, duration_seconds, average_fps, width, height, codec from videos")}
    mbps = np.array([vid_facts[v][1] * 8 / vid_facts[v][2] / 1e6 if vid_facts[v][2] else np.nan for v in names["videos"]])
    res["film_facts"] = {"mbps_p5_p50_p95": [rnd(q, 2) for q in np.nanpercentile(mbps, [5, 50, 95])], "codecs": {str(c): int(sum(1 for v in names["videos"] if vid_facts[v][6] == c)) for c in {vid_facts[v][6] for v in names["videos"]}}}
    res["shift_vs_bitrate"] = {}
    for p in POINTS:
        d = per_film(keep["u3"][p]) - per_film(keep["u1"][p])
        ok = np.isfinite(mbps)
        s = spearmanr(mbps[ok], d[ok])
        res["shift_vs_bitrate"][p] = {"spearman": rnd(s[0]), "p": rnd(s[1], 4), "films": int(ok.sum())}
    X.log(f"shift vs bitrate {res['shift_vs_bitrate']}")

    # 5. maps
    def map_row(m):
        red1, red3 = F("u1_red")[m], F("u3_red")[m]
        both, either = F("red_both")[m], F("red_either")[m]
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = both / either
        return {"frames": int(m.sum()), "pearson_median": rnd(np.nanmedian(F("pearson")[m])), "spearman_median": rnd(np.nanmedian(F("spearman")[m])),
                "abs_diff_median": rnd(np.nanmedian(F("abs_diff")[m])), "u1_red_mean": pct(np.nanmean(red1), 2), "u3_red_mean": pct(np.nanmean(red3), 2),
                "u1_red_median": pct(np.nanmedian(red1), 2), "u3_red_median": pct(np.nanmedian(red3), 2), "u1_no_red": pct((red1 == 0).mean()), "u3_no_red": pct((red3 == 0).mean()),
                "iou_median_where_any_red": rnd(np.nanmedian(iou)), "both_red_any": pct(((red1 > 0) & (red3 > 0)).mean()), "only_u1_red_any": pct(((red1 > 0) & (red3 == 0)).mean()), "only_u3_red_any": pct(((red1 == 0) & (red3 > 0)).mean()),
                "u1_mass_0": pct((F("u1_mass")[m] == 0).mean()), "u3_mass_0": pct((F("u3_mass")[m] == 0).mean()), "u1_blob_median_where_red": rnd(np.median(F("u1_blob")[m][red1 > 0]), 1), "u3_blob_median_where_red": rnd(np.median(F("u3_blob")[m][red3 > 0]), 1),
                "u1_max_median": rnd(np.nanmedian(F("u1_max")[m]), 2), "u3_max_median": rnd(np.nanmedian(F("u3_max")[m]), 2), "u1_median_cell": rnd(np.nanmedian(F("u1_median")[m]), 2), "u3_median_cell": rnd(np.nanmedian(F("u3_median")[m]), 2),
                "u1_clip_hi_frames": pct((F("u1_clip_hi")[m] > 0).mean()), "u3_clip_hi_frames": pct((F("u3_clip_hi")[m] > 0).mean()), "u1_clip_lo_cells": pct(np.nanmean(F("u1_clip_lo")[m])), "u3_clip_lo_cells": pct(np.nanmean(F("u3_clip_lo")[m]))}
    both_sharp, both_blur = keep["u1"]["r95"] & keep["u3"]["r95"], ~keep["u1"]["r95"] & ~keep["u3"]["r95"]
    res["maps"] = {"all": map_row(np.ones(N, bool)), "heldout": map_row(held), "both_sharp_r95": map_row(both_sharp), "both_blurred_r95": map_row(both_blur),
                   "only_u1_sharp_r95": map_row(keep["u1"]["r95"] & ~keep["u3"]["r95"]), "only_u3_sharp_r95": map_row(~keep["u1"]["r95"] & keep["u3"]["r95"]),
                   "zero_shots": map_row(zero_shot), "u1_sharp_r98": map_row(keep["u1"]["r98"]), "u3_sharp_r98": map_row(keep["u3"]["r98"])}
    X.log(f"maps all {res['maps']['all']}")
    # map or readout: the same map through both r, the same r on both maps
    res["readout"] = {"stored": rnd(spearmanr(L["u1"], L["u3"])[0], 4), "same_r1": rnd(spearmanr(F("u1_lme_r1"), F("u3_lme_r1"))[0], 4), "same_r3": rnd(spearmanr(F("u1_lme_r3"), F("u3_lme_r3"))[0], 4),
                      "u1_own_vs_other_r": rnd(spearmanr(F("u1_lme_r1"), F("u1_lme_r3"))[0], 4), "u3_own_vs_other_r": rnd(spearmanr(F("u3_lme_r3"), F("u3_lme_r1"))[0], 4),
                      "lme_vs_stored_median_abs": {"u1": rnd(np.median(np.abs(F("u1_lme_r1") - L["u1"]))), "u3": rnd(np.median(np.abs(F("u3_lme_r3") - L["u3"])))}}
    X.log(f"readout {res['readout']}")

    # 6a. motion at equal sharp share
    res["witness_motion_kept"] = {}
    region = motion["region"]
    for p in POINTS:
        e = {}
        for name, v in motion.items():
            ok = np.isfinite(v)
            m1 = per_film_median(v, keep["u1"][p] & ok, video, films)
            m3 = per_film_median(v, eq[p] & ok, video, films)
            e[name] = {"u1_median_kept": rnd(np.median(v[keep["u1"][p] & ok]), 3), "u3_median_kept": rnd(np.median(v[eq[p] & ok]), 3), "u3_minus_u1_films": boot(m3 - m1)}
            for lim in (4, 8, 16):
                h1 = per_film((v >= lim) & keep["u1"][p] & ok) / per_film(keep["u1"][p])
                h3 = per_film((v >= lim) & eq[p] & ok) / per_film(eq[p])
                e[name][f"share_ge_{lim}"] = {"u1": pct(((v >= lim) & keep["u1"][p]).sum() / keep["u1"][p].sum(), 2), "u3": pct(((v >= lim) & eq[p]).sum() / eq[p].sum(), 2), "u3_minus_u1_films": boot(h3 - h1)}
        e["heldout_region"] = {"u1_median_kept": rnd(np.nanmedian(region[keep["u1"][p] & held]), 3), "u3_median_kept": rnd(np.nanmedian(region[eq[p] & held]), 3)}
        res["witness_motion_kept"][p] = e
    X.log(f"witness a r95 {res['witness_motion_kept']['r95']['region']}")

    # 6b. the disputed frames: motion rank inside the shot
    res["witness_disputed"] = {}
    for p in POINTS:
        only1, only3 = keep["u1"][p] & ~eq[p], ~keep["u1"][p] & eq[p]
        m1, m3 = per_film_median(prank, only1 & np.isfinite(prank), video, films), per_film_median(prank, only3 & np.isfinite(prank), video, films)
        e = {"frames_each": int(only1.sum()), "share_of_corpus_each": pct(only1.mean(), 2), "only_u1_kept_rank": rnd(np.nanmedian(prank[only1])), "only_u3_kept_rank": rnd(np.nanmedian(prank[only3])), "u3_minus_u1_films": boot(m3 - m1)}
        for name in ("region", "subject", "camera"):
            e[f"{name}_median"] = {"only_u1_kept": rnd(np.nanmedian(motion[name][only1]), 3), "only_u3_kept": rnd(np.nanmedian(motion[name][only3]), 3)}
        hm = np.isin(np.arange(films), held_films)
        e["heldout_films"] = {"only_u1_kept_rank": [rnd(x) for x in m1[hm]], "only_u3_kept_rank": [rnd(x) for x in m3[hm]]}
        for nm, gm in (("zero_shots", zero_shot), ("other_shots", ~zero_shot & (scene >= 0))):
            e[nm] = {"only_u1_kept_rank": rnd(np.nanmedian(prank[only1 & gm])), "only_u3_kept_rank": rnd(np.nanmedian(prank[only3 & gm])), "frames_each": [int((only1 & gm).sum()), int((only3 & gm).sum())]}
        lab = code >= 0
        e["code_share"] = {"only_u1_kept": {c: pct((code[only1 & lab] == i).mean()) for i, c in enumerate(CODES)}, "only_u3_kept": {c: pct((code[only3 & lab] == i).mean()) for i, c in enumerate(CODES)}, "labelled_each": [int((only1 & lab).sum()), int((only3 & lab).sum())]}
        res["witness_disputed"][p] = e
    X.log(f"witness b r95 {res['witness_disputed']['r95']}")

    # 6c. stability: neighbouring frames of a shot
    v_sorted = video[order]
    same = sc_sorted[1:] == sc_sorted[:-1]
    same &= sc_sorted[1:] >= 0
    pairs_film = np.bincount(v_sorted[1:], same, minlength=films)
    res["witness_stability"] = {"adjacent_pairs": int(same.sum())}
    for p in POINTS:
        e = {}
        rate = {}
        for name, k in (("u1", keep["u1"][p]), ("u3_eq", eq[p]), ("u3_own", keep["u3"][p])):
            x = k[order]
            flip = (x[1:] != x[:-1]) & same
            iso = flip[1:] & flip[:-1]
            rate[name] = np.bincount(v_sorted[1:], flip, minlength=films) / pairs_film
            rate[f"{name}_iso"] = np.bincount(v_sorted[2:], iso, minlength=films) / pairs_film
            e[name] = {"flips_per_1000": rnd(flip.sum() / same.sum() * 1000, 2), "isolated_per_1000": rnd(iso.sum() / same.sum() * 1000, 2)}
        e["u3_eq_minus_u1_flips_films"] = boot((rate["u3_eq"] - rate["u1"]) * 1000)
        e["u3_eq_minus_u1_isolated_films"] = boot((rate["u3_eq_iso"] - rate["u1_iso"]) * 1000)
        res["witness_stability"][p] = e
    lag = {}
    for name, r in (("u1", r1), ("u3", r3)):
        x = r[order]
        ok = same & np.isfinite(x[1:]) & np.isfinite(x[:-1])
        lag[name] = np.array([corr(x[1:][ok & (v_sorted[1:] == f)], x[:-1][ok & (v_sorted[1:] == f)]) for f in range(films)])
    res["witness_stability"]["lag1_rank_in_shot"] = {"u1": rnd(np.median(lag["u1"])), "u3": rnd(np.median(lag["u3"])), "u3_minus_u1_films": boot(lag["u3"] - lag["u1"])}
    X.log(f"witness c {res['witness_stability']}")

    # 6d. the films released twice: every frame that can be paired
    res["witness_two_releases"] = two_releases(con, names, ids, L, th, video)
    X.log(f"witness d {res['witness_two_releases']['medians']}")

    # 6e. the cell-level motion witness
    wc1, wc3, wu1, wu3 = F("wit_comp_u1"), F("wit_comp_u3"), F("wit_unc_u1"), F("wit_unc_u3")
    has = np.isfinite(F("wit_cells"))
    moved = (F("wit_cells") >= 8) & (motion["region"] >= 4)
    res["witness_cells"] = {"frames_with_witness": int(has.sum()), "median_cells": rnd(np.nanmedian(F("wit_cells")), 0)}
    for name, m in {"all": has, "heldout": has & held, "zero_shots": has & zero_shot, "other_shots": has & ~zero_shot, "region_ge_4": moved, "both_blurred_r95": has & both_blur, "both_sharp_r95": has & both_sharp}.items():
        res["witness_cells"][name] = {"frames": int(m.sum()), "subject": {"u1": rnd(np.nanmedian(wc1[m])), "u3": rnd(np.nanmedian(wc3[m]))}, "total": {"u1": rnd(np.nanmedian(wu1[m])), "u3": rnd(np.nanmedian(wu3[m]))}}
    res["witness_cells"]["subject_u3_minus_u1_films"] = boot(per_film_median(wc3, has, video, films) - per_film_median(wc1, has, video, films))
    res["witness_cells"]["total_u3_minus_u1_films"] = boot(per_film_median(wu3, has, video, films) - per_film_median(wu1, has, video, films))
    res["witness_cells"]["region_ge_4_subject_u3_minus_u1_films"] = boot(per_film_median(wc3, moved, video, films) - per_film_median(wc1, moved, video, films))
    X.log(f"witness e {res['witness_cells']}")

    # 6f. gross check with the Qwen codes
    lab = code >= 0
    res["qwen_gross"] = {"labelled": int(lab.sum())}
    for g, gm in (("all", groups["all"]), ("heldout", held), ("rest", groups["rest"]), ("trained", groups["trained"]), ("zero_shots", zero_shot), ("other_shots", ~zero_shot & (scene >= 0))):
        m = gm & lab
        e = {"labelled": int(m.sum())}
        for p in POINTS:
            e[p] = {"sss_kept": {"u1": pct(keep["u1"][p][m & (code == 0)].mean()), "u3_own": pct(keep["u3"][p][m & (code == 0)].mean()), "u3_eq": pct(eq[p][m & (code == 0)].mean())},
                    "bbb_kept": {"u1": pct(keep["u1"][p][m & (code == 5)].mean()), "u3_own": pct(keep["u3"][p][m & (code == 5)].mean()), "u3_eq": pct(eq[p][m & (code == 5)].mean())},
                    "blurred_codes_among_kept": {"u1": pct(np.isin(code[m & keep["u1"][p]], (3, 5)).mean()), "u3_own": pct(np.isin(code[m & keep["u3"][p]], (3, 5)).mean()), "u3_eq": pct(np.isin(code[m & eq[p]], (3, 5)).mean())}}
        res["qwen_gross"][g] = e
    X.log(f"qwen {res['qwen_gross']['heldout']}")
    (X.DATA / "corpus_u1_u3.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    X.log("written")


def per_film_median(v, mask, video, films):
    mask = mask & np.isfinite(v)
    order = np.lexsort((v[mask], video[mask]))
    sorted_v, sorted_f = v[mask][order], video[mask][order]
    out = np.full(films, np.nan)
    edges = np.flatnonzero(np.r_[True, sorted_f[1:] != sorted_f[:-1], True])
    for s, e in zip(edges[:-1], edges[1:]):
        x = sorted_v[s:e]
        out[sorted_f[s]] = x[len(x) // 2] if len(x) % 2 else (x[len(x) // 2 - 1] + x[len(x) // 2]) / 2
    return out


def two_releases(con, names, ids, L, th, video):
    """As two_releases.py (time offset from the shot starts, the same frame within half a frame), on every matched frame."""
    rules = json.loads(X.RULES.read_text(encoding="utf-8"))
    vids = {}
    for vid, path, size, dur, fps in con.execute("select id, source_path, source_size_bytes, duration_seconds, average_fps from videos"):
        vids[path.replace("\\", "/").split("/")[-1]] = (vid, size * 8 / dur / 1e6, fps or 25.0)
    starts = defaultdict(list)
    for vid, st in con.execute("select video_id, start_time from scenes"):
        starts[vid].append(st)
    per = []
    for copy, kept in sorted(rules["same_film"].items()):
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
        fa = con.execute("select id, timestamp_seconds from frames where video_id = ? order by timestamp_seconds", (va,)).fetchall()
        fb = con.execute("select id, timestamp_seconds from frames where video_id = ? order by timestamp_seconds", (vb,)).fetchall()
        tb = np.array([r[1] for r in fb])
        ta = np.array([r[1] for r in fa]) + off
        j = np.searchsorted(tb, ta).clip(1, len(tb) - 1)
        jj = np.where(abs(tb[j] - ta) < abs(tb[j - 1] - ta), j, j - 1)
        ok = np.flatnonzero(abs(tb[jj] - ta) < 0.5 / fps)
        ia = np.searchsorted(ids, np.array([fa[i][0] for i in ok], "S36"))
        ib = np.searchsorted(ids, np.array([fb[jj[i]][0] for i in ok], "S36"))
        low, high = (ia, ib) if ba < bb else (ib, ia)
        e = {"pairs": int(len(ok)), "frames_a": len(fa), "frames_b": len(fb), "shot_match": rnd(best), "mbps_low": rnd(min(ba, bb), 2), "mbps_high": rnd(max(ba, bb), 2)}
        for m in L:
            lo, hi = L[m][low], L[m][high]
            e[m] = {"spearman": float(spearmanr(lo, hi)[0]), "median_low_minus_high": float(np.median(lo - hi))}
            for p in POINTS:
                agree, k = kappa(lo < th[m][p], hi < th[m][p])
                e[m][f"agree_{p}"], e[m][f"kappa_{p}"] = agree, k
        per.append(e)
    meds = {}
    for k in ("spearman", "agree_r90", "kappa_r90", "agree_r95", "kappa_r95", "agree_r98", "kappa_r98", "median_low_minus_high"):
        u1, u3 = np.array([e["u1"][k] for e in per]), np.array([e["u3"][k] for e in per])
        meds[k] = {"u1": rnd(np.median(u1), 4), "u3": rnd(np.median(u3), 4), "u3_minus_u1": boot(u3 - u1)}
    return {"films": len(per), "pairs": int(sum(e["pairs"] for e in per)), "medians": meds, "shot_match_min": min(e["shot_match"] for e in per), "per_film": per}
