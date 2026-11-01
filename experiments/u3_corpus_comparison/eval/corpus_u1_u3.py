"""Update 1 (Base, v1.0) against Update 3 (Small) over the whole corpus: 5,010,602 frames, from the scores and cell maps
stored in master.db. Plan: docs/kiserlet_u3r3_gop_letra.md, napló 2026-11-01 23:35. Read-only on master.db; no GPU.

extract: one row per frame in <cache>: frame facts, both logits, per-map summaries, the two maps compared cell by cell,
    and the cell-level motion witness of R4T-1 9.6 (cells.py) where a measurement exists.
report: the tables of the plan, from the cache, into data/corpus_u1_u3.json. Film names are not written.

usage: python experiments/u3r3_eval/corpus_u1_u3.py extract --cache DIR [--procs 16]
       python experiments/u3r3_eval/corpus_u1_u3.py report --cache DIR
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sqlite3
import time
import zlib
from pathlib import Path

import numpy as np
from scipy.ndimage import label as blobs
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
DB = "<master>/master.db"
RULES = Path("F:/RGB Mesh/dataset/frames/selection_rules_v8.json")
INDEX = ROOT / "data" / "round3_index.csv"
U1 = "4f3e943511e18e183a97623e584cdb6083754f1618fae43634ea46a048ca9e58"
U3 = "65b76e780e2110097e1ca029d05ae89be2d78f22948f59585dfd09e7787b2f72"
SETS = {"u1": U1, "u3": U3}
R = {"u1": 0.8592, "u3": 0.6953}            # solved from 300 stored maps each (spread under 0.3%)
SEED = 20261030
VALUES = np.arange(-128, 128) / 4           # the logit of each stored quarter-logit
MAP_NAMES = ("mean", "median", "p90", "p99", "max", "red", "yellow", "mass", "blob", "lme_r1", "lme_r3", "clip_hi", "clip_lo", "top1")
PAIR_NAMES = ("pearson", "spearman", "red_both", "red_either", "abs_diff")
WIT_NAMES = ("wit_cells", "wit_comp_u1", "wit_comp_u3", "wit_unc_u1", "wit_unc_u3")
FEATURES = tuple(f"{m}_{n}" for m in SETS for n in MAP_NAMES) + PAIR_NAMES + WIT_NAMES


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def connect():
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)


# ---------------------------------------------------------------------------------------------------- extraction
def map_features(v, rows, cols, stored):
    """Summaries of one stored map (flat int8 quarter logits, clipped at +-16). Red share, max and top 1% are the columns
    stored with the map, computed before rounding and clipping; the blob uses the rounded map (logit >= -0.125)."""
    n = v.size
    hist = np.bincount(v.astype(np.int16) + 128, minlength=256)
    cum = np.cumsum(hist)
    q = lambda p: VALUES[np.searchsorted(cum, p * n)]       # noqa: E731
    red = v >= 0
    inner = v.reshape(rows, cols)[1:-1, 1:-1]
    biggest = 0
    if red.any():
        lab, k = blobs(red.reshape(rows, cols))
        biggest = int(np.bincount(lab.ravel())[1:].max()) if k else 0
    out = [(hist * VALUES).sum() / n, q(0.5), q(0.9), q(0.99), stored[1], stored[0], (v >= -24).mean(),
           np.maximum(inner, 0).mean() / 4, biggest]
    for r in (R["u1"], R["u3"]):
        s = VALUES * r
        top = s[hist > 0].max()
        out.append((top + np.log((hist * np.exp(s - top)).sum() / n)) / r)
    out += [(v >= 64).mean(), (v <= -64).mean(), stored[2]]
    return out, hist, cum


def rank_table(hist, cum):
    """Average rank of each stored value (ties share a rank), so a map's ranks are one lookup."""
    return cum - (hist - 1) / 2


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return (a * b).sum() / d if d > 0 else np.nan


def witness(ev, maps):
    """R4T-1 9.6 on the stored maps: textured measured 160 px cells, the map averaged over each; within-frame Spearman."""
    mc = json.loads(zlib.decompress(ev))["motion_cells"]
    if mc["compensated"].get("status") != "measured":
        return None
    unc = {tuple(c["bbox"]): c for c in mc["uncompensated"]["cells"]}
    cells = []
    for c in mc["compensated"]["cells"]:
        if c["status"] != "measured" or c.get("structured_count", 0) < 500:
            continue
        u = unc.get(tuple(c["bbox"]))
        if not u or u.get("travel_px") is None or c.get("travel_px") is None:
            continue
        x0, y0, x1, y1 = c["bbox"]
        cells.append((x0 // 16, y0 // 16, -(-x1 // 16), -(-y1 // 16), c["travel_px"], u["travel_px"]))
    if len(cells) < 8:
        return None
    tc, tu = rankdata([c[4] for c in cells]), rankdata([c[5] for c in cells])
    out = [len(cells)]
    vs = {}
    for name in SETS:
        m = maps[name]
        vs[name] = rankdata([m[a1:b1, a0:b0].mean() for a0, a1, b0, b1, *_ in cells])
    out += [corr(vs["u1"], tc), corr(vs["u3"], tc), corr(vs["u1"], tu), corr(vs["u3"], tu)]
    return out


def work(job):
    idx, fids, rid1, rid3 = job
    con = connect()
    rows = {}
    ask = np.concatenate([rid1, rid3]).tolist()
    for start in range(0, len(ask), 2000):
        part = ",".join(map(str, ask[start:start + 2000]))
        for rid, r, c, p, share, top, top1 in con.execute(
                f"select rowid, rows, cols, payload, cells_blur_share, cell_logit_max, cell_logit_top1 from frame_blur_cells where rowid in ({part})"):
            rows[rid] = (r, c, p, (share, top, top1))
    evid = {}
    for start in range(0, len(fids), 500):
        part = fids[start:start + 500]
        for fid, p in con.execute(f"select frame_id, payload from frame_measurement_evidence where frame_id in ({','.join('?' * len(part))})", part):
            evid[fid] = p
    con.close()
    out = np.full((len(fids), len(FEATURES)), np.nan, np.float32)
    n_map = len(MAP_NAMES)
    for i, fid in enumerate(fids):
        r1, c1, p1, s1 = rows[int(rid1[i])]
        r3, c3, p3, s3 = rows[int(rid3[i])]
        a = np.frombuffer(zlib.decompress(p1), np.int8)
        b = np.frombuffer(zlib.decompress(p3), np.int8)
        fa, ha, ca = map_features(a, r1, c1, s1)
        fb, hb, cb = map_features(b, r3, c3, s3)
        out[i, :n_map] = fa
        out[i, n_map:2 * n_map] = fb
        at = 2 * n_map
        if a.size == b.size:
            ra, rb = a >= 0, b >= 0
            out[i, at:at + 5] = (corr(a.astype(np.float32), b.astype(np.float32)),
                                 corr(rank_table(ha, ca)[a.astype(np.int16) + 128], rank_table(hb, cb)[b.astype(np.int16) + 128]),
                                 (ra & rb).sum(), (ra | rb).sum(), np.abs(a.astype(np.int16) - b).mean() / 4)
        ev = evid.get(fid)
        if ev is not None:
            w = witness(ev, {"u1": a.reshape(r1, c1) / 4, "u3": b.reshape(r3, c3) / 4})
            if w is not None:
                out[i, at + 5:at + 10] = w
    return idx, out


def extract(a):
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    con = connect()
    t0 = time.time()
    log("frames")
    q = """select f.id, f.video_id, f.scene_id, f.frame_index, f.timestamp_seconds, f.blur_label, f.motion_selected,
            f.motion_branch, f.has_subtitle, f.has_overlay_text, f.visual_processing_eligible, f.width, f.height,
            json_extract(f.quality_metrics,'$.motion_region_px'), json_extract(f.quality_metrics,'$.motion_subject_px'),
            json_extract(f.quality_metrics,'$.motion_camera_px'), json_extract(f.quality_metrics,'$.motion_blur_area'),
            json_extract(f.quality_metrics,'$.shot_sharpness_rank') from frames f"""
    rows = con.execute(q).fetchall()
    log(f"frames {len(rows):,}, {time.time() - t0:.0f} s")
    order = sorted(range(len(rows)), key=lambda i: rows[i][0])
    rows = [rows[i] for i in order]
    ids = np.array([r[0] for r in rows], "S36")
    assert (ids[1:] > ids[:-1]).all()
    video_of = {}
    scene_of = {None: -1}
    v_col, s_col = np.zeros(len(rows), np.int16), np.zeros(len(rows), np.int32)
    for i, r in enumerate(rows):
        v_col[i] = video_of.setdefault(r[1], len(video_of))
        if r[2] not in scene_of:
            scene_of[r[2]] = len(scene_of) - 1
        s_col[i] = scene_of[r[2]]
    videos = sorted(video_of, key=video_of.get)
    CODES = {"sss": 0, "ssb": 1, "bss": 2, "bsb": 3, "sbb": 4, "bbb": 5}
    num = lambda k, dtype=np.float32: np.array([np.nan if r[k] is None else r[k] for r in rows], dtype)       # noqa: E731
    facts = {"video": v_col, "scene": s_col, "fidx": np.array([r[3] for r in rows], np.int32), "ts": num(4, np.float64),
             "code": np.array([CODES.get(r[5], -1) for r in rows], np.int8), "selected": np.array([r[6] == 1 for r in rows], bool),
             "teacher_branch": np.array([r[7] == "teacher-label" for r in rows], bool), "subtitle": np.array([bool(r[8]) for r in rows], bool),
             "overlay": np.array([bool(r[9]) for r in rows], bool), "eligible": np.array([bool(r[10]) for r in rows], bool),
             "width": np.array([r[11] or 0 for r in rows], np.int16), "height": np.array([r[12] or 0 for r in rows], np.int16),
             "region": num(13), "subject": num(14), "camera": num(15), "blur_area": num(16), "rank": num(17)}
    fid_strings = [r[0] for r in rows]
    del rows
    log("scores")
    logit = {m: np.full(len(ids), np.nan, np.float64) for m in SETS}
    cur = con.execute("select frame_id, set_id, logit from frame_blur_scores not indexed")
    name_of = {v: k for k, v in SETS.items()}
    while True:
        part = cur.fetchmany(1_000_000)
        if not part:
            break
        at = np.searchsorted(ids, np.array([p[0] for p in part], "S36"))
        for m in SETS:
            pick = np.array([name_of.get(p[1]) == m for p in part])
            logit[m][at[pick]] = np.array([p[2] for p, ok in zip(part, pick) if ok], np.float64)
    assert np.isfinite(logit["u1"]).all() and np.isfinite(logit["u3"]).all()
    log(f"scores done, {time.time() - t0:.0f} s")
    log("rowids")
    rid = {m: np.zeros(len(ids), np.int64) for m in SETS}
    cur = con.execute("select frame_id, set_id, rowid from frame_blur_cells indexed by sqlite_autoindex_frame_blur_cells_1")
    while True:
        part = cur.fetchmany(1_000_000)
        if not part:
            break
        at = np.searchsorted(ids, np.array([p[0] for p in part], "S36"))
        for m in SETS:
            pick = np.array([name_of.get(p[1]) == m for p in part])
            rid[m][at[pick]] = np.array([p[2] for p, ok in zip(part, pick) if ok], np.int64)
    assert (rid["u1"] > 0).all() and (rid["u3"] > 0).all()
    log(f"rowids done, {time.time() - t0:.0f} s")
    settings = dict(con.execute("select key, value from settings where key = 'blur_model_active_set'").fetchall())
    meta = {m: json.loads(con.execute("select meta from blur_model_sets where set_id = ?", (s,)).fetchone()[0])["thresholds"] for m, s in SETS.items()}
    scene_ids = sorted((k for k in scene_of if k is not None), key=scene_of.get)
    con.close()
    np.save(cache / "ids.npy", ids)
    np.savez(cache / "facts.npz", **facts, logit_u1=logit["u1"], logit_u3=logit["u3"])
    (cache / "names.json").write_text(json.dumps({"videos": videos, "scenes": scene_ids, "thresholds": meta, "active": settings}), encoding="utf-8")

    feats = np.lib.format.open_memmap(cache / "features.npy", mode="w+", dtype=np.float32, shape=(len(ids), len(FEATURES)))
    walk = np.argsort(rid["u1"])
    jobs = ((walk[s:s + a.chunk], [fid_strings[i] for i in walk[s:s + a.chunk]], rid["u1"][walk[s:s + a.chunk]], rid["u3"][walk[s:s + a.chunk]])
            for s in range(0, len(walk), a.chunk))
    done = 0
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for idx, out in pool.imap_unordered(work, jobs):
            feats[idx] = out
            done += len(idx)
            if done % (a.chunk * 50) < a.chunk:
                rate = done / (time.time() - t0)
                log(f"  {done:,} / {len(ids):,}, {rate:,.0f} frames/s overall")
    feats.flush()
    (cache / "features.json").write_text(json.dumps(FEATURES), encoding="utf-8")
    log(f"extract done, {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("extract", "report"))
    ap.add_argument("--cache", required=True)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=4000)
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a)
    else:
        import corpus_u1_u3_report as rep
        rep.report(a)


if __name__ == "__main__":
    main()
