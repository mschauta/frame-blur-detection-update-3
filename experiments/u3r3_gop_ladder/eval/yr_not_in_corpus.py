"""Is the test series training material? Checked 2026-11-01, before its name was published (napló).

Every model of the project trains on the frames of the corpus (master.db, read-only). Three checks against its videos:
- by name: the source paths of the videos and their assets;
- by length: a corpus video within 2 s or 48 frames of one of the 18 episodes;
- by content: the full SHA-256 of the 18 AVC files against the corpus digests (the corpus is H.264 only, so the 10-bit
  files cannot be in it).

usage: python experiments/u3r3_eval/yr_not_in_corpus.py
Output: data/yr_not_in_corpus.json
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

import numpy as np

import common as C
from yr import VIDEOS

MASTER = Path("<master>")
NAME = re.compile(r"young.{0,3}royals|szerelem|k.teless.g|S0[1-3]E0[1-6]", re.I)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def main():
    con = sqlite3.connect(f"file:{MASTER / 'master.db'}?mode=ro", uri=True)
    videos = con.execute("SELECT content_sha256, source_path, reported_frame_count, duration_seconds FROM videos").fetchall()
    assets = con.execute("SELECT relative_path, absolute_path FROM source_assets").fetchall()
    avc = {lab: path for lab, path in VIDEOS.items() if lab.endswith("_AVC")}
    frames = {lab: len(np.load(C.DATA / f"yr_{lab}.npz", allow_pickle=False)["type"]) for lab in avc}
    digests = {v[0] for v in videos}
    res = {"corpus_videos": len(videos), "corpus_assets": len(assets),
           "name_hits": sum(bool(NAME.search(v[1] or "")) for v in videos) + sum(bool(NAME.search(a or "")) for row in assets for a in row),
           "same_length_within_2s": sum(1 for n in frames.values() for v in videos
                                        if (v[2] and abs(v[2] - n) <= 48) or (v[3] and abs(v[3] - n / 24) <= 2)),
           "avc_files_checked": len(avc), "avc_files_in_corpus": sum(sha256(p) in digests for p in avc.values())}
    (C.DATA / "yr_not_in_corpus.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    C.log(str(res))


if __name__ == "__main__":
    main()
