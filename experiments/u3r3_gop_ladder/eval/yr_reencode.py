"""Is Update 3's blur reading on the AVC episode the low-bitrate x264 coding? One minute of S01E02, re-encoded.

Reference: the HEVC Main 10 file decoded (the item-5 pipeline). From the same decoded frames (converted to 8 bit
4:2:0 or 10 bit 4:2:0), three encodes:
- x264 at 2.7 Mbit/s ABR, preset slow, keyint 240, 3 B-frames (the AVC file's rate, a film-like GOP);
- x264 CRF 18, otherwise the same (x264 at high quality);
- x265 10 bit at 2.7 Mbit/s ABR, preset medium, keyint 240.
Each is decoded and scored with Update 3 and the round3 Small; PSNR against the reference frames. The original AVC
file's stored scores of the same frames, matched by the luma sequence, as the comparison.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from yr import VIDEOS

W, H, T0, DUR = 1920, 1080, 600.0, 60.0
SRC = VIDEOS["S01E02_HEVC"]
DEC = ["-vf", "scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24", "-f", "rawvideo", "-"]
VARIANTS_FIRST = {
    "x264_2.7M": ["-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "slow", "-b:v", "2700k", "-maxrate", "4500k",
                  "-bufsize", "9000k", "-g", "240", "-bf", "3"],
    "x264_crf18": ["-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-g", "240", "-bf", "3"],
    "x265_10bit_2.7M": ["-pix_fmt", "yuv420p10le", "-c:v", "libx265", "-preset", "medium", "-b:v", "2700k",
                        "-x265-params", "keyint=240:vbv-maxrate=4500:vbv-bufsize=9000:log-level=error"],
}
# second pass (added after the first): lossless 4:2:0 round trips, 8 and 10 bit, and lossless x264 8 bit
VARIANTS = {
    "ffv1_yuv420p_lossless": ["-pix_fmt", "yuv420p", "-c:v", "ffv1"],
    "ffv1_yuv420p10_lossless": ["-pix_fmt", "yuv420p10le", "-c:v", "ffv1"],
    "x264_qp0_lossless": ["-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-qp", "0"],
    "x264_crf12": ["-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-g", "240", "-bf", "3"],
}


def decode(path, ss=None, dur=None):
    cmd = ["ffmpeg", "-v", "error"] + (["-ss", str(ss)] if ss is not None else []) + ["-i", str(path)]
    cmd += (["-t", str(dur)] if dur is not None else []) + ["-map", "0:v:0", "-fps_mode", "passthrough"] + DEC
    p = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(p, np.uint8).reshape(-1, H, W, 3)


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


def main():
    u3, mu = C.load_u3()
    sm, _ = C.load_small()
    models = {"u3": u3, "small": sm}
    ref = decode(SRC, T0, DUR)
    n = len(ref)
    res = {"HEVC (reference)": score(models, ref)}
    tmp = Path(tempfile.mkdtemp())
    ps = {}
    for name, args in {**VARIANTS_FIRST, **VARIANTS}.items():
        out = tmp / f"{name}.mkv"
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24",
                                "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv", *args, str(out)], stdin=subprocess.PIPE)
        enc.stdin.write(ref.tobytes())
        enc.stdin.close()
        enc.wait()
        fr = decode(out)
        assert len(fr) == n, (name, len(fr), n)
        res[name] = score(models, fr)
        ps[name] = psnr(fr, ref)
        kbps = out.stat().st_size * 8 / DUR / 1000
        print(f"{name}: {kbps:.0f} kbit/s, PSNR vs reference {ps[name]:.2f} dB", flush=True)
    # the original AVC file's stored scores of the same frames
    st = np.load(C.DATA / "yr_S01E02_AVC.npz")
    sh = np.load(C.DATA / "yr_S01E02_HEVC.npz")
    wl = np.array([0.2126, 0.7152, 0.0722], np.float32)
    luma = np.array([(f[60:H - 60].astype(np.float32) @ wl).mean() for f in ref])
    c0 = int(T0 * 24)
    s0 = min(range(c0 - 48, c0 + 48), key=lambda s: np.abs(sh["luma"][s:s + n] - luma).mean())
    print(f"aligned to stored frame {s0}; stored HEVC U3 vs rescored max |d| {np.abs(sh['u3_logit'][s0:s0 + n] - res['HEVC (reference)']['u3']).max():.4f}")
    res["AVC file (stored)"] = {"u3": st["u3_logit"][s0:s0 + n], "small": st["small_logit"][s0:s0 + n]}
    r = res["HEVC (reference)"]
    for name, e in res.items():
        print(f"{name:20s} U3 px median {np.median(C.measured_L(e['u3'], mu)):5.2f}, logit {np.median(e['u3']):6.2f}, Spearman with reference {spearmanr(e['u3'], r['u3'])[0]:.3f} "
              f"| Small logit {np.median(e['small']):6.2f}, Spearman with reference {spearmanr(e['small'], r['small'])[0]:.3f}")


if __name__ == "__main__":
    main()
