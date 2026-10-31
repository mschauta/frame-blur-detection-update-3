"""Protocol item 5: one series in several codecs (the test series is not named in anything published).

Scope: the protocol registers all 18 episodes in AVC and HEVC SDR and the 12 in AV1 SDR. The first round (2026-10-30)
ran S01E01 in AVC and HEVC SDR (the R4K-1 video A, with the flash and strobe scenes of napló 13:05) and S01E02 in AVC,
HEVC SDR and AV1 SDR (the first episode that exists in all three SDR codecs); that subset was a deviation from the
protocol (napló 2026-10-31 02:24). The full round (from 2026-10-31) runs the rest, episode by episode. Every frame is
scored with Update 3 and the round3 Small (published Update 1 loader, bfloat16 autocast, whole 1920 x 1080 frame with
its bars, as the FrameBlurStudio scores). Frame types, packet sizes and times from ffprobe; the mean luma of the
active picture (rows 60 ... 1019) per frame.

Measures (protocol): the agreement of the scores between codecs on the same frame (Spearman of the logits; the share
of frames whose r95 decision differs), and the Phase II I-frame enrichment (the sharpest frame of each 1 / 5 / 20 s
window, ties shared evenly, black frames excluded, against the I-frame share of the windows; R4K-1 Q6). Added after
napló 13:05: light jumps (luma >= 1.4 x the median of +-4 frames) by frame type, as control. Over the episodes
(recorded before the full round, napló 2026-10-31): the median and range of each measure per episode, and Update 3
minus the Small with a bootstrap over the episodes.

usage: python experiments/u3r3_eval/yr.py score [--only LABEL]     (resumable per video)
       python experiments/u3r3_eval/yr.py report
Output: data/yr_<label>.npz, data/yr.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import queue
import time
from pathlib import Path

import numpy as np
import torch

import common as C

VID = Path("F:/FrameBlurStudio/input video")
CODECS = {"AVC": "AVC_SDR", "HEVC": "HEVC_SDR", "AV1": "AV1_SDR"}
W, H, BATCH = 1920, 1080, 8


def _videos() -> dict:
    """label -> file, episode by episode (AVC, HEVC, AV1), so that a stopped run leaves whole episodes."""
    found = {}
    for codec, folder in CODECS.items():
        for path in sorted((VID / folder).glob("*.mkv")):
            found[f"{re.search(r'S[0-9]{2}E[0-9]{2}', path.name).group(0)}_{codec}"] = path
    return dict(sorted(found.items(), key=lambda kv: (kv[0].split("_")[0], list(CODECS).index(kv[0].split("_")[1]))))


VIDEOS = _videos()


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "frame=pict_type,pkt_size,best_effort_timestamp_time,key_frame", "-of", "json", str(path)],
                         capture_output=True, check=True).stdout
    fr = json.loads(out)["frames"]
    return {"type": np.array([f["pict_type"] for f in fr]), "key": np.array([int(f["key_frame"]) for f in fr]),
            "pkt_size": np.array([int(f.get("pkt_size", 0)) for f in fr]),
            "t": np.array([float(f.get("best_effort_timestamp_time", "nan")) for f in fr])}


def frames(path: Path, q: queue.Queue):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough",
                          "-vf", "scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24",
                          "-f", "rawvideo", "-"], stdout=subprocess.PIPE, bufsize=W * H * 3 * BATCH)
    n = W * H * 3
    while True:
        buf = p.stdout.read(n * BATCH)
        if not buf:
            break
        k = len(buf) // n
        q.put(np.frombuffer(buf[:k * n], np.uint8).reshape(k, H, W, 3))
    p.wait()
    q.put(None)


def score_video(label: str, path: Path, models: dict, meta_u3: dict):
    dst = C.DATA / f"yr_{label}.npz"
    if dst.exists():
        C.log(f"{label}: már kész")
        return
    pr = {}
    th = threading.Thread(target=lambda: pr.update(probe(path)))
    th.start()
    q: queue.Queue = queue.Queue(maxsize=6)
    dec = threading.Thread(target=frames, args=(path, q), daemon=True)
    dec.start()
    out = {m: [] for m in models}
    luma, t0, n = [], time.time(), 0
    wl = torch.tensor([0.2126, 0.7152, 0.0722], device="cuda").view(1, 3, 1, 1)
    while True:
        b = q.get()
        if b is None:
            break
        x = torch.from_numpy(b).permute(0, 3, 1, 2).contiguous().cuda()
        for m, model in models.items():
            out[m].append(C.forward(model, x)["image"].float().cpu().numpy())
        luma.append((x[:, :, 60:H - 60].float() * wl).sum(1).mean((1, 2)).cpu().numpy())
        n += len(b)
        if n % 6000 < BATCH:
            C.log(f"  {label}: {n} frame, {n / (time.time() - t0):.0f} frame/s")
    th.join()
    res = {f"{m}_logit": np.concatenate(v) for m, v in out.items()}
    res["luma"] = np.concatenate(luma)
    assert len(res["luma"]) == len(pr["type"]), (label, len(res["luma"]), len(pr["type"]))
    res["u3_L"] = C.measured_L(res["u3_logit"], meta_u3)
    np.savez_compressed(dst, **res, **{k: v for k, v in pr.items()}, video=str(path), sha256_head=C.sha256_head(path))
    C.log(f"{label}: {n} frame, {time.time() - t0:.0f} s; I-frame {np.mean(pr['type'] == 'I'):.2%}")


def score(a):
    C.DATA.mkdir(parents=True, exist_ok=True)
    u3, mu = C.load_u3()
    sm, _ = C.load_small()
    models = {"u3": u3, "small": sm}
    failed = []
    for label, path in VIDEOS.items():
        if a.only and label != a.only:
            continue
        try:
            score_video(label, path, models, mu)
        except Exception as exc:                # one bad file must not stop a run of many hours
            failed.append(label)
            C.log(f"{label}: HIBA, kihagyva: {exc!r}")
    if failed:
        C.log(f"hibás videók ({len(failed)}): {failed}")
        sys.exit(1)
    C.log("minden videó kész")


def _nb_median(x, w=4):
    n = len(x)
    pad = np.pad(x, w, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1)
    win = np.delete(win, w, axis=1)
    return np.median(win, 1)[:n]


def phase2(t, typ, score_, eligible, rng, reps=2000):
    out = {}
    for width in (1, 5, 20):
        window = np.floor((t - t[0]) / width).astype(int)
        picked, expected = [], []
        for w in np.unique(window[eligible]):
            cand = np.flatnonzero((window == w) & eligible)
            v = score_[cand]
            best = cand[v == v.min()]
            picked.append(np.mean(typ[best] == "I"))
            expected.append(np.mean(typ[cand] == "I"))
        picked, expected = np.array(picked), np.array(expected)
        boot = []
        for _ in range(reps):
            b = rng.integers(0, len(picked), len(picked))
            boot.append(picked[b].mean() / max(expected[b].mean(), 1e-12))
        out[f"{width}s"] = {"windows": int(len(picked)), "i_share_of_picks": float(picked.mean()),
                            "i_share_expected": float(expected.mean()), "enrichment": float(picked.mean() / expected.mean()),
                            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    return out


FIRST_ROUND = ("S01E01", "S01E02")


def _summary(per_ep: dict, rng, reps=2000) -> dict:
    """episode -> {u3, small}: the median and range per model, U3 - Small with a bootstrap over the episodes, and
    whether the first round's episodes lie inside the range of the others."""
    eps = sorted(per_ep)
    out = {"episodes": len(eps)}
    for m in ("u3", "small"):
        v = np.array([per_ep[ep][m] for ep in eps], float)
        out[m] = {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}
        rest = [per_ep[ep][m] for ep in eps if ep not in FIRST_ROUND]
        first = [per_ep[ep][m] for ep in eps if ep in FIRST_ROUND]
        if rest and first:
            out[m]["first_round_inside_range_of_others"] = bool(min(rest) <= min(first) and max(first) <= max(rest))
    d = np.array([per_ep[ep]["u3"] - per_ep[ep]["small"] for ep in eps], float)
    boot = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(reps)]
    out["u3_minus_small"] = {"median": float(np.median(d)), "ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
                             "u3_higher_episodes": int((d > 0).sum())}
    return out


def over_episodes(res: dict, rng) -> dict:
    """The episode is the unit here. All of it is one series, so the intervals say nothing about other material."""
    out = {"note": "intervals: bootstrap over the episodes of one series"}
    for pair in ("AVC~HEVC", "AVC~AV1", "HEVC~AV1"):
        x, y = pair.split("~")
        items = {k.split("_")[0]: e for k, e in res["agreement"].items()
                 if k.endswith(f"_{y}") and f"_{x}~" in k and "error" not in e}
        if not items:
            continue
        for name, key in (("spearman_logits", "spearman_logits"), ("decision_flips_r95", "decision_flips_r95"),
                          ("median_logit_x_minus_y", "median_logit_x_minus_y")):
            out[f"{pair}/{name}"] = _summary({ep: {m: e[m][key] for m in ("u3", "small")} for ep, e in items.items()}, rng)
    for codec in CODECS:
        videos = {k.split("_")[0]: e for k, e in res["videos"].items() if k.endswith(f"_{codec}")}
        if not videos:
            continue
        for window in ("1s", "5s", "20s"):
            out[f"{codec}/enrichment_{window}"] = _summary(
                {ep: {m: e[m]["phase2"][window]["enrichment"] for m in ("u3", "small")} for ep, e in videos.items()}, rng)
        px = np.array([e["u3_px_median"] for e in videos.values()])
        out[f"{codec}/u3_px_median"] = {"episodes": len(px), "median": float(np.median(px)), "min": float(px.min()), "max": float(px.max())}
    return out


def report(a):
    _, mu = C.load_u3("cpu")
    _, ms = C.load_small("cpu")
    th = {"u3": C.logit_threshold(mu, "r95"), "small": C.logit_threshold(ms, "r95")}
    D = {lab: dict(np.load(C.DATA / f"yr_{lab}.npz", allow_pickle=False)) for lab in VIDEOS if (C.DATA / f"yr_{lab}.npz").exists()}
    rng = np.random.default_rng(C.SEED)
    res = {"thresholds_logit_r95": th, "videos": {}, "agreement": {}}
    for lab, d in D.items():
        typ, t = d["type"], d["t"]
        t = np.where(np.isfinite(t), t, np.arange(len(t)) / 24.0)
        nb = _nb_median(d["luma"])
        lit = (d["luma"] >= 8) & (nb >= 8)
        spike = lit & (d["luma"] >= 1.4 * nb) & (d["luma"] < 230)
        e = {"frames": int(len(typ)), "i_share": float(np.mean(typ[lit] == "I")), "black_or_dark_excluded": int((~lit).sum())}
        for m in ("u3", "small"):
            s = d[f"{m}_logit"]
            e[m] = {"phase2": phase2(t, typ, s, lit, rng),
                    "sharp_r95_by_type": {ty: float(np.mean(s[lit & (typ == ty)] < th[m])) for ty in ("I", "P", "B") if (lit & (typ == ty)).any()},
                    "light_jump": {g: {"n": int(sel.sum()), "sharp_r95": float(np.mean(s[sel] < th[m])) if sel.any() else None,
                                       "median_logit_minus_neighbours": float(np.median(s[sel] - _nb_median(s)[sel])) if sel.any() else None}
                                   for g, sel in (("I_with_jump", spike & (typ == "I")), ("I_without_jump", lit & ~spike & (typ == "I")),
                                                  ("P_with_jump", spike & (typ == "P")), ("P_without_jump", lit & ~spike & (typ == "P")),
                                                  ("B", lit & (typ == "B")))}}
        e["u3_px_median_by_type"] = {ty: float(np.median(d["u3_L"][lit & (typ == ty)])) for ty in ("I", "P", "B") if (lit & (typ == ty)).any()}
        e["u3_px_median"] = float(np.median(d["u3_L"][lit]))
        e["u3_logit_median"] = float(np.median(d["u3_logit"][lit]))
        e["small_logit_median"] = float(np.median(d["small_logit"][lit]))
        res["videos"][lab] = e
    from scipy.stats import spearmanr
    for ep in sorted({lab.split("_")[0] for lab in D}):
        labs = [lab for lab in D if lab.startswith(ep)]
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                x, y = D[labs[i]], D[labs[j]]
                if len(x["type"]) != len(y["type"]):
                    res["agreement"][f"{labs[i]}~{labs[j]}"] = {"error": "frame counts differ"}
                    continue
                lit = (x["luma"] >= 8) & (y["luma"] >= 8)
                e = {"frames": int(lit.sum()), "same_type_share": float(np.mean(x["type"][lit] == y["type"][lit]))}
                for m in ("u3", "small"):
                    sx, sy = x[f"{m}_logit"][lit], y[f"{m}_logit"][lit]
                    cx, cy = sx >= th[m], sy >= th[m]
                    both_non_i = (x["type"][lit] != "I") & (y["type"][lit] != "I")
                    e[m] = {"spearman_logits": float(spearmanr(sx, sy)[0]), "decision_flips_r95": float(np.mean(cx != cy)),
                            "decision_flips_r95_non_I": float(np.mean(cx[both_non_i] != cy[both_non_i])),
                            "median_abs_logit_difference": float(np.median(np.abs(sx - sy))),
                            "median_logit_x_minus_y": float(np.median(sx - sy))}
                res["agreement"][f"{labs[i]}~{labs[j]}"] = e
    res["over_episodes"] = over_episodes(res, rng)
    (C.DATA / "yr.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for lab, e in res["videos"].items():
        C.log(f"{lab}: I-arány {e['i_share']:.2%}; II. fázis dúsulás 1/5/20 s: "
              + "; ".join(f"{m} " + " / ".join(f"{e[m]['phase2'][w]['enrichment']:.1f}" for w in ("1s", "5s", "20s")) for m in ("u3", "small"))
              + " | fényugrás I/P éles: " + "; ".join(f"{m} {e[m]['light_jump']['I_with_jump']['sharp_r95']} / {e[m]['light_jump']['P_with_jump']['sharp_r95']}" for m in ("u3", "small")))
    for k, e in res["agreement"].items():
        if "error" in e:
            C.log(f"{k}: {e['error']}")
            continue
        C.log(f"{k}: " + "; ".join(f"{m} Spearman {e[m]['spearman_logits']:.3f}, döntésváltás r95 {e[m]['decision_flips_r95']:.2%}" for m in ("u3", "small")))
    for k, e in res["over_episodes"].items():
        if "u3_minus_small" in e:
            d = e["u3_minus_small"]
            C.log(f"{k} ({e['episodes']} epizód): U3 {e['u3']['median']:.3f} [{e['u3']['min']:.3f}–{e['u3']['max']:.3f}], Small {e['small']['median']:.3f} "
                  f"[{e['small']['min']:.3f}–{e['small']['max']:.3f}]; U3 − Small {d['median']:+.3f} {np.round(d['ci95'], 3).tolist()}, "
                  f"első kör a többi tartományán belül: U3 {e['u3'].get('first_round_inside_range_of_others')}, Small {e['small'].get('first_round_inside_range_of_others')}")
        elif "median" in e:
            C.log(f"{k} ({e['episodes']} epizód): {e['median']:.3f} [{e['min']:.3f}–{e['max']:.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("score", "report"))
    ap.add_argument("--only")
    a = ap.parse_args()
    {"score": score, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
