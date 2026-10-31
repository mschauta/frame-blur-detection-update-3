"""Update 3 pre-check: are the VLM labels (and the round3 Base scores) of the J: frames biased by the coded frame type?

For every J: video, the packets of the source file (I:\delete, matched by name and size) give the keyframe flag per
presentation timestamp; J:'s frames carry the source pts. Inside a scene, an I-frame is compared with its two
neighbours (frame index - 1 and + 1, same scene, all three labelled): is the I-frame's Qwen code sharper, blurrier or
the same? The null is the same comparison centred on non-I frames (a fixed-seed sample). The round3 Base corpus score
(1 - blur_model_low = blur probability) is compared the same way.

Read-only on master.db. usage: python experiments/u3_precheck/label_frametype.py  -> data/label_frametype.json
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DB = "<master>/master.db"
SRC = Path("I:/delete")
SHARPNESS = {"sss": 3, "ssb": 2, "bss": 2, "sbb": 1, "bsb": 1, "bbb": 0}     # number of "sharp" answers


def keyframes(path: Path) -> tuple[set[int], int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts,flags",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.split()
    keys, n = set(), 0
    for line in out:
        a = line.split(",")
        if len(a) >= 2 and a[0] not in ("", "N/A"):
            n += 1
            if "K" in a[1]:
                keys.add(int(a[0]))
    return keys, n


def main():
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    vids = db.execute("SELECT id, source_path, source_size_bytes FROM videos").fetchall()
    files = {p.name: p for p in SRC.iterdir()}
    src = {}
    for vid, sp, size in vids:
        f = files.get(Path(sp).name)
        assert f is not None and f.stat().st_size == size, sp
        src[vid] = f
    with ThreadPoolExecutor(10) as ex:
        kf = dict(zip(src, ex.map(keyframes, src.values())))
    rows = db.execute("SELECT video_id, frame_index, pts, scene_id, blur_label, blur_model_low FROM frames "
                      "WHERE blur_label IN ('sss','ssb','bss','sbb','bsb','bbb')").fetchall()
    by = {}
    for vid, fi, pts, sc, lab, low in rows:
        by[(vid, fi)] = (pts, sc, SHARPNESS[lab], None if low is None else 1 - low)
    matched = sum(pts in kf[vid][0] for (vid, _), (pts, *_r) in by.items())
    res = {"videos": len(src), "labelled_frames": len(by), "labelled_keyframes": int(matched)}
    rng = np.random.default_rng(20261029)
    trip = defaultdict(list)
    for (vid, fi), (pts, sc, s, p) in by.items():
        a, b = by.get((vid, fi - 1)), by.get((vid, fi + 1))
        if not a or not b or a[1] != sc or b[1] != sc:
            continue
        kind = "I" if pts in kf[vid][0] else "nonI"
        if kind == "nonI" and rng.random() > 0.05:      # 5% sample of the non-I centres as the null
            continue
        trip[kind].append((s, a[2], b[2], p, a[3], b[3]))
    for kind, t in trip.items():
        t = np.array([[x if x is not None else np.nan for x in r] for r in t], float)
        s, n1, n2, p, p1, p2 = t.T
        nb = (n1 + n2) / 2
        ok = np.isfinite(p) & np.isfinite(p1) & np.isfinite(p2)
        res[kind] = {
            "triples": int(len(t)),
            "qwen_centre_sharper_than_both": float(np.mean((s > n1) & (s > n2))),
            "qwen_centre_blurrier_than_both": float(np.mean((s < n1) & (s < n2))),
            "qwen_all_three_equal": float(np.mean((s == n1) & (s == n2))),
            "qwen_mean_sharpness_centre_minus_neighbours": float(np.mean(s - nb)),
            "qwen_share_sss_centre": float(np.mean(s == 3)), "qwen_share_sss_neighbours": float(np.mean((n1 == 3) | False) / 2 + np.mean(n2 == 3) / 2),
            "base_triples": int(ok.sum()),
            "base_centre_lowest_blur_prob": float(np.mean((p < p1) & (p < p2) & ok) / max(ok.mean(), 1e-9)),
            "base_median_logit_centre_minus_neighbours": float(np.median(
                (np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1)) -
                 (np.log(np.clip(p1, 1e-9, 1 - 1e-9) / np.clip(1 - p1, 1e-9, 1)) +
                  np.log(np.clip(p2, 1e-9, 1 - 1e-9) / np.clip(1 - p2, 1e-9, 1))) / 2)[ok])),
        }
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "label_frametype.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
