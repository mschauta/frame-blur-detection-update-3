"""Added to item 5 after the full round (napló 2026-10-31): what the files of the test series are, and how the readings
spread inside a file. Nothing here was registered; it describes the material behind the registered measures.

- AVC: the x264 version and settings from the banner the encoder leaves at the start of the stream.
- AV1: the film-grain flags in the sequence and frame headers (ffmpeg trace_headers, the whole file).
- Every file: the video rate and the median packet size per frame type (ffprobe, stored by yr.py).
- Every file, lit frames: the share of frames Update 3 reads over 1 px, its median px per frame type, the quartiles of
  both models' logits, the Spearman of the two models inside the file, and Update 3's median px per decile of Small.

- `pairs`, per episode: Update 3's px, AVC minus HEVC, as the gap of the two medians and as the paired per-frame median.

usage: python experiments/u3r3_eval/yr_file_facts.py
       python experiments/u3r3_eval/yr_file_facts.py pairs     (adds the px gaps to the stored file)
Output: data/yr_file_facts.json
"""
from __future__ import annotations

import json
import re
import subprocess
import sys

import numpy as np
from scipy.stats import spearmanr

import common as C
from yr import VIDEOS

X264 = re.compile(rb"x264 - core (\d+) (r\d+ \w+) - H\.264/MPEG-4 AVC codec - Copy[^\x00]*?options: ([^\x00]+)")
# The settings that matter for the reading, or that differ between the files.
X264_KEYS = ("crf", "rc", "ref", "bframes", "b_pyramid", "weightp", "keyint", "keyint_min", "scenecut", "rc_lookahead", "mbtree", "qcomp",
             "aq", "psy_rd", "deblock", "vbv_maxrate", "vbv_bufsize", "me", "subme", "me_range", "trellis", "threads")


def x264_settings(path) -> dict | None:
    with open(path, "rb") as f:
        m = X264.search(f.read(8 << 20))
    if not m:
        return None
    opts = dict(kv.split("=", 1) for kv in m.group(3).decode("latin1").split() if "=" in kv)
    return {"core": m.group(1).decode(), "build": m.group(2).decode(), **{k: opts.get(k) for k in X264_KEYS}}


def grain_flags(path) -> dict:
    p = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "info", "-i", str(path), "-map", "0:v:0", "-c", "copy",
                          "-bsf:v", "trace_headers", "-f", "null", "-"], stderr=subprocess.PIPE, text=True, errors="replace")
    out = {"film_grain_params_present": {"0": 0, "1": 0}, "apply_grain": {"0": 0, "1": 0}}
    for line in p.stderr:
        m = re.search(r" (film_grain_params_present|apply_grain)\s+\S+\s+= (\d)", line)
        if m:
            out[m.group(1)][m.group(2)] += 1
    p.wait()
    return out


def file_row(label, d) -> dict:
    typ, pkt = d["type"], d["pkt_size"].astype(np.float64)
    lit = d["luma"] >= 8
    L, u, s, ty = d["u3_L"][lit], d["u3_logit"][lit], d["small_logit"][lit], typ[lit]
    edges = np.percentile(s, np.arange(10, 100, 10))
    bins = np.digitize(s, edges)
    return {"video_mbit_s": float(pkt.sum() * 8 / (len(typ) / 24) / 1e6),
            "packet_kb_median": {t: float(np.median(pkt[typ == t]) / 1024) for t in "IPB" if (typ == t).any()},
            "u3_over_1px": float(np.mean(L > 1)),
            "u3_px_median_by_type": {t: float(np.median(L[ty == t])) for t in "IPB" if (ty == t).any()},
            "u3_logit_quartiles": np.percentile(u, [25, 50, 75]).tolist(),
            "small_logit_quartiles": np.percentile(s, [25, 50, 75]).tolist(),
            "spearman_u3_small_inside_file": float(spearmanr(u, s)[0]),
            "u3_px_median_by_small_decile": [float(np.median(L[bins == i])) for i in range(10)]}


def pairs():
    path = C.DATA / "yr_file_facts.json"
    res = json.loads(path.read_text(encoding="utf-8"))
    for ep in sorted({lab.split("_")[0] for lab in VIDEOS}):
        a, h = [dict(np.load(C.DATA / f"yr_{ep}_{c}.npz", allow_pickle=False)) for c in ("AVC", "HEVC")]
        lit = (a["luma"] >= 8) & (h["luma"] >= 8)
        res[f"{ep}_AVC"]["u3_px_minus_hevc"] = {"gap_of_medians": float(np.median(a["u3_L"][lit]) - np.median(h["u3_L"][lit])),
                                                "paired_median": float(np.median(a["u3_L"][lit] - h["u3_L"][lit]))}
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "pairs":
        return pairs()
    res = {}
    for label, path in VIDEOS.items():
        d = dict(np.load(C.DATA / f"yr_{label}.npz", allow_pickle=False))
        e = file_row(label, d)
        codec = label.split("_")[1]
        if codec == "AVC":
            e["x264"] = x264_settings(path)
        if codec == "AV1":
            e["grain"] = grain_flags(path)
        res[label] = e
        C.log(f"{label}: {e['video_mbit_s']:.2f} Mbit/s, 1 px fölött {e['u3_over_1px']:.0%}, "
              f"{e.get('x264', {}).get('crf') if codec == 'AVC' else e.get('grain', '')}")
    (C.DATA / "yr_file_facts.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
