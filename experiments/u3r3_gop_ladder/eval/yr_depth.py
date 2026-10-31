"""Added to item 5 (napló 2026-10-31): does the re-encoding effect belong to the codec or to the bit depth?

The re-encoding test (yr_reencode.py) ran x264 in 8 bit and x265 in 10 bit, so the two were not separated. Here the same
minute of the episode 2 HEVC file (600-660 s, decoded to 8-bit RGB as the reference) goes through a 2 x 2 design:
x264 / x265 x 8 / 10 bit, with the encoder settings of the first pass unchanged (ABR 2.7 Mbit/s, VBV 4.5 / 9 Mbit,
keyint 240; x264 preset slow, 3 B-frames; x265 preset medium). Each is decoded and scored with Update 3 and the round3
Small. x264 8 bit and x265 10 bit repeat the first pass, as a control.

Measures per model, on the paired per-frame logits (and px for Update 3):
  codec effect  = median of ((x264_8 - x265_8) + (x264_10 - x265_10)) / 2
  depth effect  = median of ((x264_8 - x264_10) + (x265_8 - x265_10)) / 2
  interaction   = median of (x264_8 - x264_10) - (x265_8 - x265_10)
with a block bootstrap over 2 s blocks (2000 resamples, seed 20261030). The decision rule is in the log, recorded
before the run: "codec" if |codec| >= 2 |depth| and its interval excludes 0; "depth" the other way round; else
"not separated".

Limits: one minute, second-generation encodes, and the input is 8-bit RGB, so "10 bit" is the encoder's precision, not
a 10-bit source as in the real files.

usage: python experiments/u3r3_eval/yr_depth.py
       python experiments/u3r3_eval/yr_depth.py simple     (added after the decision, from the stored logits)
Output: data/yr_depth.npz, data/yr_depth.json, data/yr_depth_simple.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

import common as C
from yr import VIDEOS

W, H, T0, DUR, BLOCK = 1920, 1080, 600.0, 60.0, 48
SRC = VIDEOS["S01E02_HEVC"]
DEC = ["-vf", "scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24", "-f", "rawvideo", "-"]
X264 = ["-c:v", "libx264", "-preset", "slow", "-b:v", "2700k", "-maxrate", "4500k", "-bufsize", "9000k", "-g", "240", "-bf", "3"]
X265 = ["-c:v", "libx265", "-preset", "medium", "-b:v", "2700k",
        "-x265-params", "keyint=240:vbv-maxrate=4500:vbv-bufsize=9000:log-level=error"]
PIX = {8: "yuv420p", 10: "yuv420p10le"}
VARIANTS = {f"{name}_{depth}": ["-pix_fmt", PIX[depth], *args]
            for name, args in (("x264", X264), ("x265", X265)) for depth in (8, 10)}
# The first pass, for the control (data/yr_reencode.log).
FIRST_PASS = {"x264_8": {"u3_px": 3.04, "small_logit": 3.36}, "x265_10": {"u3_px": 1.72, "small_logit": 0.41}}


def decode(path, ss=None, dur=None):
    cmd = ["ffmpeg", "-v", "error"] + (["-ss", str(ss)] if ss is not None else []) + ["-i", str(path)]
    cmd += (["-t", str(dur)] if dur is not None else []) + ["-map", "0:v:0", "-fps_mode", "passthrough"] + DEC
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.uint8).reshape(-1, H, W, 3)


def score(models, fr):
    out = {m: [] for m in models}
    for i in range(0, len(fr), 8):
        x = torch.from_numpy(np.ascontiguousarray(fr[i:i + 8])).permute(0, 3, 1, 2).contiguous().cuda()
        for m, model in models.items():
            out[m].append(C.forward(model, x)["image"].float().cpu().numpy())
    return {m: np.concatenate(v) for m, v in out.items()}


def psnr(a, b):
    mse = np.array([np.mean((x.astype(np.float32) - y) ** 2) for x, y in zip(a, b)])
    return float(np.median(10 * np.log10(255 ** 2 / np.maximum(mse, 1e-9))))


def effects(v, rng, reps=2000):
    """Main effects and interaction on paired per-frame values, with a block bootstrap."""
    a8, a10, b8, b10 = v["x264_8"], v["x264_10"], v["x265_8"], v["x265_10"]
    per = {"codec": ((a8 - b8) + (a10 - b10)) / 2, "depth": ((a8 - a10) + (b8 - b10)) / 2,
           "interaction": (a8 - a10) - (b8 - b10)}
    n = len(a8) // BLOCK
    starts = np.arange(n) * BLOCK
    res = {}
    for k, d in per.items():
        boot = []
        for _ in range(reps):
            ix = np.concatenate([np.arange(s, s + BLOCK) for s in rng.choice(starts, n)])
            boot.append(np.median(d[ix]))
        res[k] = {"median": float(np.median(d)), "ci95": np.percentile(boot, [2.5, 97.5]).tolist()}
    return res


def verdict(e):
    c, d = abs(e["codec"]["median"]), abs(e["depth"]["median"])
    excl = lambda k: e[k]["ci95"][0] > 0 or e[k]["ci95"][1] < 0            # noqa: E731
    if c >= 2 * d and excl("codec"):
        return "codec"
    if d >= 2 * c and excl("depth"):
        return "depth"
    return "not separated"


def simple():
    """Added after the decision: the effect inside each codec and bit depth, and each variant against the reference."""
    d = np.load(C.DATA / "yr_depth.npz")
    _, mu = C.load_u3("cpu")
    rng = np.random.default_rng(C.SEED)
    n = 1440 // BLOCK
    starts = np.arange(n) * BLOCK

    def med(x):
        boot = [np.median(x[np.concatenate([np.arange(s, s + BLOCK) for s in rng.choice(starts, n)])]) for _ in range(2000)]
        return {"median": float(np.median(x)), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}

    def row(ua, ub, sa, sb):
        return {"u3_logit": med(ua - ub), "u3_px": med(C.measured_L(ua, mu) - C.measured_L(ub, mu)), "small_logit": med(sa - sb)}

    get = lambda k, m: d[f"{k}_{m}"].astype(np.float64)                     # noqa: E731
    res = {"pairs": {f"{a} - {b}": row(get(a, "u3"), get(b, "u3"), get(a, "small"), get(b, "small"))
                     for a, b in (("x264_8", "x264_10"), ("x265_8", "x265_10"), ("x264_8", "x265_8"), ("x264_10", "x265_10"))},
           "against_reference": {k: row(get(k, "u3"), d["ref_u3"].astype(np.float64), get(k, "small"), d["ref_small"].astype(np.float64))
                                 for k in VARIANTS}}
    (C.DATA / "yr_depth_simple.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for group in res.values():
        for k, e in group.items():
            C.log(k + ": " + "; ".join(f"{m} {v['median']:+.2f} {np.round(v['ci95'], 2).tolist()}" for m, v in e.items()))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "simple":
        return simple()
    u3, mu = C.load_u3()
    sm, _ = C.load_small()
    models = {"u3": u3, "small": sm}
    ref = decode(SRC, T0, DUR)
    C.log(f"referencia: {len(ref)} frame")
    ref_scores = score(models, ref)
    logits, table = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, args in VARIANTS.items():
            out = Path(tmp) / f"{name}.mkv"
            enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24",
                                    "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv", *args, str(out)], stdin=subprocess.PIPE)
            enc.stdin.write(ref.data)
            enc.stdin.close()
            assert enc.wait() == 0, name
            info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,profile,pix_fmt",
                                   "-of", "csv=p=0", str(out)], capture_output=True, text=True, check=True).stdout.strip()
            fr = decode(out)
            assert len(fr) == len(ref), (name, len(fr), len(ref))
            s = score(models, fr)
            logits[name] = s
            table[name] = {"stream": info, "kbit_s": out.stat().st_size * 8 / DUR / 1000, "psnr_to_reference_db": psnr(fr, ref),
                           "u3_px_median": float(np.median(C.measured_L(s["u3"], mu))), "u3_logit_median": float(np.median(s["u3"])),
                           "small_logit_median": float(np.median(s["small"])),
                           "u3_spearman_with_reference": float(spearmanr(s["u3"], ref_scores["u3"])[0]),
                           "small_spearman_with_reference": float(spearmanr(s["small"], ref_scores["small"])[0])}
            e = table[name]
            C.log(f"{name} ({info}): {e['kbit_s']:.0f} kbit/s, PSNR {e['psnr_to_reference_db']:.2f} dB | U3 {e['u3_px_median']:.2f} px, "
                  f"logit {e['u3_logit_median']:+.2f}, Spearman {e['u3_spearman_with_reference']:.3f} | "
                  f"Small logit {e['small_logit_median']:+.2f}, Spearman {e['small_spearman_with_reference']:.3f}")
            del fr
    rng = np.random.default_rng(C.SEED)
    series = {"u3_logit": {k: logits[k]["u3"].astype(np.float64) for k in VARIANTS},
              "u3_px": {k: C.measured_L(logits[k]["u3"], mu) for k in VARIANTS},
              "small_logit": {k: logits[k]["small"].astype(np.float64) for k in VARIANTS}}
    eff = {name: effects(v, rng) for name, v in series.items()}
    res = {"what": __doc__.split("\n\n")[0], "minute_s": [T0, T0 + DUR], "frames": len(ref),
           "reference": {"u3_px_median": float(np.median(C.measured_L(ref_scores["u3"], mu))),
                         "small_logit_median": float(np.median(ref_scores["small"]))},
           "variants": table, "effects": eff,
           "verdict_by_decision_rule": {"u3": verdict(eff["u3_logit"]), "small": verdict(eff["small_logit"])},
           "control_first_pass": {k: {"first_pass": v, "now": {"u3_px": table[k]["u3_px_median"], "small_logit": table[k]["small_logit_median"]}}
                                  for k, v in FIRST_PASS.items()}}
    (C.DATA / "yr_depth.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez_compressed(C.DATA / "yr_depth.npz", ref_u3=ref_scores["u3"], ref_small=ref_scores["small"],
                        **{f"{k}_{m}": logits[k][m] for k in VARIANTS for m in models})
    C.log(f"referencia: U3 {res['reference']['u3_px_median']:.2f} px, Small logit {res['reference']['small_logit_median']:+.2f}")
    for k, v in eff.items():
        C.log(f"{k}: " + "; ".join(f"{e} {v[e]['median']:+.3f} {np.round(v[e]['ci95'], 3).tolist()}" for e in v))
    C.log(f"döntés a rögzített szabállyal: {res['verdict_by_decision_rule']}")
    C.log(f"kontroll az első menethez: {res['control_first_pass']}")


if __name__ == "__main__":
    main()
