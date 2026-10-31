# Update 3 report: the GOP-coded continuous ladder, epoch 29

The protocol is [PROTOCOL.md](PROTOCOL.md) (English translation) and [PROTOCOL_hu.md](PROTOCOL_hu.md) (the registered
text, SHA-256 in [PROTOCOL.sha256](PROTOCOL.sha256)). Every measure of the protocol is reported below, with the two
deviations listed under the summary. Analyses added after the first results are marked **added**, with the result that
prompted them.

The control is Update 1 Small (release v1.1, `round3_small_p99_ep029.pt`): the same frames, the same start, the same
schedule. Both files were scored through the published Update 1 loader (`blurdetect`), bfloat16 autocast, fingerprint
in float32, no mask, on one RTX 5090.

The scripts as run are in [eval/](eval/) and their aggregate outputs in [data/](data/). The publication edits and
the hashes are in [MANIFEST.json](MANIFEST.json). The scripts need the non-public corpus and the unpublished training
code, so they do not rerun elsewhere. This report translates the protocol log's entries from the end of the run
onwards. The entries written during training are summarised in the README as the course of the run.

## Summary

| item | Update 3 | Update 1 Small | proposed criterion |
|---|---|---|---|
| 1. R4K-2 codec gap (P/B − I-only, logit; x264 CRF 23, pan) | +0.00 | +13.20 | **met** in all 7 cells where Small has a gap |
| 2. held-out AUC, six films | 0.9627 | 0.9620 | **met:** +0.0007 [−0.0017, +0.0028] |
| 3(a). pair order 0–3 px, Update 1 ladder | 0.996 | 0.659 | **met** |
| 3(b). px MAE / Spearman, 3 codecs × I-only / IPB × static / pan | 1.47–1.52 px / 0.990–0.995 | (no px) | – |
| 4. cell map against motion, within the frame | 0.249 | 0.338 | – (worse) |
| 5. agreement of the AVC and the HEVC encodes of the same frames (Spearman; 18 episodes, median and range) | 0.429 (0.290–0.569) | 0.862 (0.778–0.902) | – (worse) |
| 5. r95 decisions that change between the AVC and the HEVC encode (18 episodes) | 29.7% (22.8–42.3) | 20.6% (16.3–24.0) | – (worse) |
| 5. agreement of the HEVC and the AV1 encodes (Spearman; 12 episodes) | 0.752 (0.633–0.841) | 0.650 (0.577–0.913) | – (better) |
| 5. Phase II I-frame enrichment, AVC, 5 s windows (18 episodes) | 15.8× (10.7–25.4) | 69.6× (56.7–81.9) | – |
| 6. `bbb` frames under 1 px, held-out films | 5.6% | (no px) | R4T-1: 17–29% |

- **On photo clips the frame-type bias is gone.** Update 3 reads an I-frame and the P/B frames of the same clip the
  same, in all three codecs. Where the I-only coding is worse (lower PSNR), it reads the I-only frames as blurrier: it
  follows the coding quality, not the frame type.
- **The frame decision is not inferior by the protocol's margin, and no gain is shown.** Sharp frames kept are level
  at r90 and r95, and lower at r98 (−2.13 [−3.13, −1.22] points; section 3).
- **The px scale holds on photos** in every codec and structure, and on native frames the blurred VLM codes no longer
  read under 1 px (5.6% of `bbb`). Inside a code, the cell readouts order the frames by measured motion as well as
  Update 1 Small's. This does not validate the px as a physical length on video: the order of the codes is partly
  met by construction, steady pans read almost 0 px, and the reading changes with the encode.
- **On film (18 episodes of one series) the frame-type bias is reduced, not removed, and the encode still matters.**
  - **The I-frame preference.** On the AVC files, which are x264 at CRF 16–17, the share called sharp at r95 is nearly
    level across the frame types: I 25%, P 19%, B 23% (Small: I 84%, P 22%, B 28%). When picking the sharpest frame
    per 5 s the I-frame enrichment falls to about a quarter of Small's (15.8× against 69.6×), and is still large.
  - **AVC against HEVC.** In all 18 episodes both models read the AVC file blurrier: by +2.78 logits for Update 3 (+1.00
    to +5.57) and +1.81 for Small (+1.50 to +2.17). Update 3 ranks the frames of the two much less alike (Spearman 0.429
    against 0.862) and changes more r95 decisions (29.7% against 20.6%).
  - **HEVC against AV1 Update 3 is the steadier** (Spearman 0.752 against 0.650; decisions 17.1% against 27.0%). One AV1
    file without film grain suggests that grain, not the codec, lowers Small's agreement there.
  - **The px depend on the file, not only the codec.** Update 3's median on AVC ranges from 0.21 to 3.15 px over the
    episodes (HEVC 0.03–0.23). Episodes 13–17 and 18 read 0.21–0.44 and 2.28 px, with the same HEVC reading and the same
    x264 settings apart from the CRF. The cause is not established.
  - **The two episodes of the first round were not typical,** by a rule recorded before the full round. The direction
    holds, but they overstated the difference in level and ranking, and understated the decision changes.
  - In a one-minute re-encoding test, x264 coding of film frames raised the reading in steps with the quality, by
    0.6 px even at 43 dB, while x265 at the same rate raised it less. A follow-up 2 × 2 test points to the codec for
    Update 1 Small. For Update 3 it does not separate the codec from the bit depth. The encodes are second generation,
    from an HEVC source (section 6).
  - Update 1 Small reacts the same way in logits, and its decision crosses the boundary; it keeps its ranking
    better.
  - On two H.264 releases of 31 corpus films, Update 3 agrees as well as Small (Spearman 0.801 against 0.793).
- **The cell map localises motion less well** inside a frame than Update 1 Small's (0.249 against 0.338).
- **The 95% point is very strict.** Its threshold sits at 0.05 px, so it calls 21% of coded 0 px photos blurred
  (Update 1 Small: 4%), as strict as on the sharp validation frames.

One run, three changes (photo coding, continuous ladder, no masks): no result here can be assigned to one of them.

## Deviations from the protocol

- **Item 5 ran in two rounds.** The protocol registers 18 episodes in AVC and HEVC and 12 in AV1.
  - The first round ran episodes 1 and 2 in AVC and HEVC, and episode 2 in AV1. They were chosen on 2026-10-30, before
    the item 5 results: episode 1 for its flash scenes, episode 2 as the first in all three codecs. Those results were
    published, and the re-encoding tests of section 6 were designed from them.
  - The full round ran on 2026-10-31 (03:41–23:25), with its analysis recorded in the log before it started. Section 6
    reports all 18 episodes.
  - The log wrongly called the first round's subset a choice within the protocol. The protocol fixes the episodes too.
- **Item 4's visual inspection has not been done.** Only the cell-level witness was run.

## 1. Run and export

- **The run** finished on 2026-10-30 at 14:27: 30 epochs, 90,509 steps, 117,448 s. The checkpoint fixed in advance is
  epoch 29.
- **Validation frames, epoch 29** (Update 1 Small epoch 29 in brackets): AUC 0.9617 (0.9612); balanced accuracy
  0.8947 (0.8926); balanced NLL 0.3301 (0.3231); sharp kept at r90 / r95 / r98: 88.7 / 79.0 / 63.7% (87.9 / 77.7 /
  64.1%).
- **Over all 30 epochs** the frame AUC was above Update 1 Small's in 24 (mean 0.9477 against 0.9434); in epochs 20–29
  in 9 of 10 (0.9603 against 0.9592).
- **Validation photos, epoch 29:** Spearman 0.992, MAE 1.43 px; the frame decision flips at 1.21 px of true photo
  length (0.80–1.30 px over epochs 20–29). At 0 px the median reading is 0.02 px on I targets and 0.01 px on P/B
  targets; per codec the two differ by at most 0.01 px.
- **Export:** `update3_small_p99_ep029.pt`, 136,223,051 bytes, SHA-256
  `6cd68d831b63fdc5f8a6c02cef777c89fbd273daed4d3f47a3c134a37421fbf2`. The state dict without the two ordinal
  parameters, so the Update 1 loader reads it unchanged; `meta["ordinal"]` holds the grid T and the learned offsets b.
  - Thresholds: r90 0.2362, r95 0.0321, r98 0.0078 (on the px scale: 0.355, 0.052 and 0.013 px).
  - b for T = 0.5 … 30 px: −0.73, 0.19, 1.79, 3.64, 5.59, 7.61, 9.67, 11.76, 13.95, 16.01, 18.17, 20.32, 22.53, 25.69,
    29.07, 36.25. The frame boundary s = 0 therefore reads 0.9 px.
  - Parity on a probe frame between the training model and the exported file read by the Update 1 loader: image
    logit −0.58192 in both, cell difference 0, px 0.5284 in both.

## 2. Item 1: codec dependence on R4K-2

All 115,200 R4K-2 frames were coded again with R4K-2's own code. Every frame matched the stored row: frame type and
packet size exactly, PSNR to within 5·10⁻⁴ dB. Update 1 Small, scored again, matched its stored scores to within
5·10⁻⁵ logit.

The codec gap: the paired logit difference of the P/B frames (IP and IPB) against the I-only coding of the same
source frame, per motion and rate (95% interval of the Update 3 / Small ratio, R4K-2's photo resamples):

| motion / rate | Update 3 | Small | Update 3 / Small | blurred-share gap at r95, U3 / Small |
|---|---|---|---|---|
| static / CRF 23 | +0.14 | +12.84 | 0.01 [0.00, 0.02] | +0.036 / +0.053 |
| static / CRF 28 | −0.91 | +4.80 | −0.19 [−0.25, −0.14] | −0.089 / 0.000 |
| static / CRF 33 | −1.91 | −2.15 | 0.89 [0.68, 1.20] | −0.394 / 0.000 |
| static / 2 Mbit/s | −3.31 | +8.77 | −0.38 [−0.45, −0.31] | −0.631 / +0.080 |
| pan / CRF 23 | +0.00 | +13.20 | 0.00 [−0.01, 0.01] | −0.017 / +0.045 |
| pan / CRF 28 | −0.94 | +11.53 | −0.08 [−0.10, −0.06] | −0.099 / +0.025 |
| pan / CRF 33 | −1.69 | +8.09 | −0.21 [−0.25, −0.17] | −0.369 / +0.044 |
| pan / 2 Mbit/s | −3.32 | +9.20 | −0.36 [−0.42, −0.31] | −0.628 / +0.047 |

- **The proposed criterion is met.** In all 7 cells where Small has a positive gap, the ratio is at most 0.01 and its
  upper bound at most 0.024; the interval of the difference excludes 0 in all 7. In static / CRF 33 Small has no
  positive gap either (−2.15), so there is nothing to halve.
- **The sign turns under strong compression:** Update 3 reads the I-only frames as blurrier (static, 2 Mbit/s: 84.8%
  called blurred at r95 against 21% of the P/B frames). The I-only frames are worse: PSNR 32.1 dB against 40.4–40.7
  dB at 2 Mbit/s, 33.7 against 37.0 dB at CRF 33.
- **Inside one clip** (IP, IPB) the I- and the P/B frames read the same: median 0.011–0.015 px, and 18–23% called
  blurred.
- **R4K-1 repeated** (is the in-clip I-frame sharper than both neighbours): Update 3 0.13–0.24, Small 0.36–1.00; the
  chance level in pan clips is 0.30–0.35. Update 3 no longer prefers the I-frame. Within a clip the P/B frames are
  0.2–0.4 dB better than their I-frame (static), 0.0–0.4 dB (pan).
- **False alarms on the 0 px photos** (every R4K-2 frame is 0 px):

  | | r90 | r95 | r98 |
  |---|---|---|---|
  | Update 3, IP + IPB | 5.9% | 20.7% | 49.8% |
  | Small, IP + IPB | 0.6% | 3.6% | 7.7% |
  | Update 3, I-only | 17.7% | 47.9% | 78.9% |
  | Small, I-only | 0.0% | 0.0% | 0.1% |

  The px themselves are small (IP + IPB: median 0.013 px, 1.8% at 1 px or more; I-only: 9.3%, at 2 Mbit/s the 90th
  percentile is 2.0 px). The thresholds are what is strict: r95 sits at 0.052 px. Update 3 keeps 79.3% of these
  photos at r95, as it keeps 79.0% of the sharp validation frames; Update 1 Small is far more lenient on photos than
  on film (96.4% against 77.7%).

## 3. Item 2: held-out films

The six held-out films of R4S-1 (114,687 VLM-labelled frames); intervals from 2000 bootstrap resamples over the shots,
the same resamples for both models.

| | Update 3 | Small | difference |
|---|---|---|---|
| AUC, six films | 0.9627 | 0.9620 | +0.0007 [−0.0017, +0.0028] |
| AUC, three validation films | 0.9584 | 0.9578 | +0.0006 [−0.0014, +0.0028] |
| AUC, three test films | 0.9659 | 0.9654 | +0.0005 [−0.0034, +0.0036] |
| sharp kept at r90, six films | 89.0% | 88.8% | +0.17 [−0.21, +0.52] points |
| sharp kept at r95 | 78.9% | 79.0% | −0.10 [−0.65, +0.43] points |
| sharp kept at r98 | 63.6% | 65.8% | −2.13 [−3.13, −1.22] points |
| Spearman with the motion measurement | 0.704 | 0.705 | [−0.008, +0.007] |

- **The proposed criterion is met:** the interval of the AUC difference does not cross −0.005.
- Per film Update 3 is above Small on three films and below on three (largest differences +0.0019 and −0.0008).
- Each file's operating points come from its own validation thresholds. At r95 both catch exactly the same number of
  blurred frames (41,546 of 43,677); at r98 Update 3 catches 61 more.
- On the test films alone: r95 −1.16 [−2.03, −0.42], r98 −3.15 [−4.71, −1.78] points; on the validation films +0.88
  [+0.16, +1.63] and −1.20 [−2.47, +0.03].
- The two agree on 94.9% of the r95 decisions (κ 0.90, Spearman of the logits 0.962); in the 5,814 disagreements the
  VLM label sides with Update 3 in 49.4%.

## 4. Item 3: photo ladder

**(a) The Update 1 ladder** (300 photos, 17 lengths, single-frame H.264 CRF 23):

| | Update 3 | Small |
|---|---|---|
| pair order 0–3 px | **0.996** | 0.659 |
| pair order 3–16 px | 1.000 | 0.992 |
| strictly increasing over 0–1–2–3 px | **0.997** | 0.253 |
| 50% flip at r90 / r95 / r98 | 1.01 / 0.54 / 0.12 px | 9.10 / 9.04 / 8.99 px |

- Update 3's median reading at L = 0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30 px: 0.01, 0.05,
  0.34, 0.95, 1.45, 1.91, 2.40, 3.37, 4.37, 5.36, 7.04, 8.46, 10.18, 13.89, 16.49, 20.46, 23.48 px. MAE 1.49 px,
  Spearman 0.993. Short lengths read about 0.6 px short; the top reads short (30 px → 23.5 px).
- **The proposed criterion is met.** This ladder is close to the training photos (single-frame x264); the transfer to
  film is decided by items 5 and 6.

**(b) The same photos in GOP clips** (the first 100 photos × 17 lengths × x264 CRF 23 / x265 CRF 26 / AV1 CRF 38 ×
I-only and IPB × static and 2 px/frame pan along the blur, 8 frames, no grain; 20,400 clips, 163,200 frames):

| codec / structure | PSNR (dB) | MAE (0–3 px) | Spearman | at 0 px: median px | at 0 px: blurred at r95, U3 / Small |
|---|---|---|---|---|---|
| x264, I-only | 39.2–39.3 | 1.49 (0.49–0.50) | 0.994 | 0.01 | 14–25% / 0.1% |
| x264, IPB | 40.7–40.9 | 1.48–1.50 (0.44–0.48) | 0.995 | 0.01–0.02 | 12–21% / 0–1% |
| x265, I-only | 38.0 | 1.50 (0.51–0.52) | 0.994 | 0.01 | 16–19% / 0% |
| x265, IPB | 39.3–39.4 | 1.47–1.49 (0.44–0.48) | 0.995 | 0.01 | 8–19% / 0% |
| AV1, I-only | 36.6 | 1.51–1.52 (0.55–0.56) | 0.990 | 0.02 | 25–26% / 2% |
| AV1, IPB | 39.4–39.7 | 1.47–1.48 (0.45–0.48) | 0.995 | 0.01 | 7–17% / 0% |

(The ranges span the static and the pan clip, and for IPB the I- and the P/B frames.)

- **No codec or frame-type dependence of the px:** MAE 1.47–1.52 px and Spearman 0.990–0.995 in all 18 groups.
- **Inside a clip** (IPB, P/B frames against their own I-frame, median): Update 3 −0.00 to +0.04 logit, at most
  0.015 px. Small: +4.55 logit in x264 pans (+5.55 at 0 px), −0.15 to +0.07 elsewhere.
- **Against the I-only coding** (IPB P/B − I-only, per photo and length): Update 3 −0.03 to +0.03; Small x264 +1.73
  (static) and +7.31 (pan), x265 −0.04 / −0.17, AV1 −0.64 / −0.71. In this kind of clip Small's P/B bias is specific
  to x264; R4K-2 was x264 only.
- The AV1 encoder crashed three times (access violation); each time the first retry succeeded.

## 5. Item 4: cell maps

**The cell-level motion witness** (R4T-1 9.6: the same 2,000 validation frames, 137,934 textured measured 160 px
cells; the within-frame Spearman between the cell map and the measured motion, median):

| | Update 3 | Small | round3 Base (R4T-1) |
|---|---|---|---|
| subject motion (compensated) | **0.249** [0.232, 0.266] | 0.338 [0.321, 0.350] | 0.339 |
| – frames not close-up / close-up | 0.226 / 0.294 | 0.297 / 0.404 | |
| total motion | 0.247 | 0.329 | 0.332 |
| static detail | −0.173 | −0.159 | −0.179 |
| pooled over all cells | 0.460 | 0.512 | 0.507 |

- **Update 3's cell map ranks the cells of a frame by motion less well:** −0.076 [−0.085, −0.068] against Small.
  Which of the three changes causes it cannot be told: the ladder (on a photo the whole picture is blurred alike, so
  the cell target is the same L everywhere), the coding, or the dropped watermark mask.

**The cell px over training** (median; image px / inner cells' 90th percentile):

| epoch | sss | bsb | bbb | pan 4–8 | 8–16 | 16–24 | 24+ px/frame |
|---|---|---|---|---|---|---|---|
| 0 | 1.88 / 1.29 | 4.71 / 3.24 | 8.03 / 6.87 | 2.35 / 1.47 | 3.39 / 2.78 | 6.56 / 6.23 | 10.18 / 9.67 |
| 12 | 0.02 / 0.02 | 2.71 / 1.96 | 5.57 / 5.92 | 0.05 / 0.03 | 0.11 / 0.12 | 2.06 / 2.79 | 5.70 / 6.33 |
| 15 | 0.01 / 0.01 | 1.63 / 0.97 | 3.55 / 4.05 | 0.03 / 0.03 | 0.07 / 0.10 | 0.99 / 1.66 | 3.87 / 4.62 |
| 29 | 0.00 / 0.01 | 2.27 / 1.60 | 4.02 / 4.62 | 0.01 / 0.01 | 0.03 / 0.04 | 0.72 / 1.03 | 3.57 / 4.48 |

- The 4,347 steady camera pans of the validation frames, by VLM code: 4–8 px/frame 63% `sss`, 9% `bbb`; 8–16: 59%
  and 13%; 16–24: 36% and 36%; 24+: 2% and 81%.
- Epoch 29 follows the labels: the pans labelled `sss` read 0.00–0.06 px (90–97% under 1 px), those labelled `bbb`
  3.23 / 1.82 / 2.64 / 3.90 px in the four bands.
- **The cells do not hide blur on pans:** at 4–16 px/frame even the largest inner cell reads only 1.4–1.5 px (median).
- Epoch 0 read everything blurrier and still ranked the pans by speed; by epoch 12 the `sss` pans read 0 px.
- The author's visual inspection on known films needs Update 3 in FrameBlurStudio; it is not done yet.

## 6. Item 5: one series in several codecs

Every frame of the 18 episodes in AVC and HEVC and of the 12 in AV1, as registered: 48 files, 3,207,491 frames of the
test series (not named), scored with both models; the whole 1920 × 1080 frame with its bars, as FrameBlurStudio scores.

- **The episodes are numbered 1–18 in order.** AV1 exists for all but episodes 1 and 13–17.
- **The files of one episode align frame by frame.** In all 42 codec pairs the luma sequences correlate best at zero
  offset, at 0.9999 or more.
- **The unit is the episode.** The intervals are bootstraps over the episodes (2000 resamples, seed 20261030). All of it
  is one series, so they say nothing about other material.

| codec | format | video rate, median (range) | encoder |
|---|---|---|---|
| AVC | H.264 High, 8 bit | 2.14 (1.77–2.61) Mbit/s | x264 core 148: CRF 16, or 17 in episodes 1, 12 and 18 |
| HEVC | Main 10, I and B frames | 1.78 (1.47–2.48) Mbit/s | not known |
| AV1 | Main, 10 bit | 1.17 (0.53–1.60) Mbit/s | not known; film-grain synthesis in 11 of the 12 files |

- **A correction.** An earlier version of this report said the encoders of these files were not known. The AVC files
  carry the x264 banner at the start of the stream; it had not been read. In all 18 files the rate control and coding
  settings are the same apart from the CRF; episode 1 also has a newer x264 build and another thread count
  ([data/yr_file_facts.json](data/yr_file_facts.json)):
  `rc=crf, ref=4, bframes=16, b_pyramid=2, weightp=2, keyint=240, keyint_min=121, scenecut=0, rc_lookahead=240,
  mbtree=1, qcomp=0.50, aq=1:0.80, psy_rd=1.00:0.00, deblock=1:0:0, vbv_maxrate=20000, vbv_bufsize=25000, me=umh,
  subme=10, me_range=120, trellis=2`.
- **Film grain.** In 11 AV1 files `apply_grain` is set in 92–95% of the frame headers, and the decoder used here applies
  it. Episode 7's AV1 file has no film grain at all.

**Agreement between the encodes, over the episodes** (Spearman of the logits on the same frames; the share of frames
whose r95 decision differs; the median paired logit difference):

| measure | Update 3, median [range] | Small, median [range] | U3 − Small [95%] | U3 higher in |
|---|---|---|---|---|
| AVC ~ HEVC, Spearman | 0.429 [0.290, 0.569] | 0.862 [0.778, 0.902] | −0.433 [−0.463, −0.404] | 0 / 18 |
| AVC ~ HEVC, r95 decisions that change | 29.7% [22.8, 42.3] | 20.6% [16.3, 24.0] | +12.4 [+4.5, +14.9] | 18 / 18 |
| AVC ~ HEVC, median logit, AVC − HEVC | +2.78 [+1.00, +5.57] | +1.81 [+1.50, +2.17] | +0.96 [+0.25, +1.64] | 14 / 18 |
| AVC ~ AV1, Spearman | 0.450 [0.316, 0.564] | 0.622 [0.494, 0.864] | −0.177 [−0.218, −0.156] | 0 / 12 |
| AVC ~ AV1, r95 decisions that change | 28.5% [22.7, 37.0] | 28.7% [20.3, 37.9] | −0.7 [−3.1, +2.3] | 5 / 12 |
| AVC ~ AV1, median logit, AVC − AV1 | +2.93 [+0.70, +4.34] | +1.93 [+0.99, +2.43] | +1.11 [+0.59, +2.04] | 10 / 12 |
| HEVC ~ AV1, Spearman | 0.752 [0.633, 0.841] | 0.650 [0.577, 0.913] | +0.114 [+0.045, +0.155] | 10 / 12 |
| HEVC ~ AV1, r95 decisions that change | 17.1% [15.6, 23.4] | 27.0% [12.4, 31.0] | −9.3 [−11.0, −7.6] | 1 / 12 |
| HEVC ~ AV1, median logit, HEVC − AV1 | +0.14 [−1.21, +0.50] | +0.03 [−0.60, +0.80] | +0.11 [−0.24, +0.67] | 6 / 12 |

**Phase II I-frame enrichment** (the sharpest frame of each window, ties shared, dark frames excluded, against the
I-frame share of the windows):

| encode, window | Update 3, median [range] | Small, median [range] | U3 − Small [95%] | U3 higher in |
|---|---|---|---|---|
| AVC, 1 s | 7.4 [5.3, 10.0] | 19.8 [16.4, 21.3] | −12.4 [−12.9, −11.2] | 0 / 18 |
| AVC, 5 s | 15.8 [10.7, 25.4] | 69.6 [56.7, 81.9] | −53.7 [−57.5, −49.7] | 0 / 18 |
| AVC, 20 s | 10.8 [4.2, 20.5] | 91.7 [81.0, 109.8] | −83.2 [−90.0, −75.3] | 0 / 18 |
| HEVC, 1 s | 2.7 [1.9, 3.9] | 1.5 [1.0, 2.3] | +1.3 [+0.9, +1.5] | 18 / 18 |
| HEVC, 5 s | 4.5 [2.3, 7.7] | 1.8 [0.5, 4.3] | +2.7 [+2.0, +3.2] | 18 / 18 |
| HEVC, 20 s | 5.3 [2.6, 15.7] | 2.5 [0.0, 7.5] | +3.1 [+1.2, +4.6] | 16 / 18 |
| AV1, 1 s | 2.5 [1.8, 4.2] | 2.1 [1.0, 2.7] | +0.9 [+0.3, +1.2] | 11 / 12 |
| AV1, 5 s | 3.4 [1.5, 8.5] | 2.9 [0.8, 5.0] | +0.7 [−0.6, +2.3] | 8 / 12 |
| AV1, 20 s | 3.4 [0.8, 16.3] | 4.4 [0.0, 9.4] | +0.2 [−2.3, +0.8] | 6 / 12 |

**Episode by episode:**

| episode | x264 CRF | U3 px, AVC / HEVC / AV1 | AVC frames over 1 px | Spearman AVC ~ HEVC, U3 / Small | logit AVC − HEVC, U3 / Small | r95 changes, U3 / Small | 5 s enrichment on AVC, U3 / Small |
|---|---|---|---|---|---|---|---|
| 1 | 17 | 3.15 / 0.15 / – | 73% | 0.328 / 0.828 | +5.57 / +2.17 | 25.5% / 20.6% | 24.2 / 69.4 |
| 2 | 16 | 2.77 / 0.19 / 0.21 | 68% | 0.290 / 0.778 | +4.20 / +2.09 | 24.6% / 24.0% | 23.0 / 81.8 |
| 3 | 16 | 2.83 / 0.19 / 0.21 | 72% | 0.379 / 0.860 | +4.49 / +2.15 | 22.8% / 22.2% | 20.6 / 75.7 |
| 4 | 16 | 2.25 / 0.23 / 0.19 | 63% | 0.379 / 0.823 | +2.77 / +1.76 | 24.2% / 20.6% | 19.9 / 81.9 |
| 5 | 16 | 2.36 / 0.18 / 0.15 | 64% | 0.413 / 0.853 | +3.24 / +1.98 | 24.6% / 21.1% | 15.8 / 78.9 |
| 6 | 16 | 1.32 / 0.16 / 0.11 | 53% | 0.456 / 0.881 | +1.71 / +1.64 | 26.6% / 18.9% | 10.7 / 77.2 |
| 7 | 16 | 2.54 / 0.11 / 0.48 | 65% | 0.498 / 0.902 | +3.96 / +1.81 | 29.6% / 16.8% | 15.8 / 69.0 |
| 8 | 16 | 2.30 / 0.11 / 0.11 | 62% | 0.511 / 0.891 | +2.98 / +1.81 | 29.3% / 17.3% | 13.0 / 67.3 |
| 9 | 16 | 2.08 / 0.15 / 0.20 | 61% | 0.459 / 0.845 | +2.39 / +1.95 | 26.7% / 22.5% | 13.1 / 69.7 |
| 10 | 16 | 2.62 / 0.17 / 0.15 | 66% | 0.404 / 0.891 | +2.80 / +1.89 | 29.9% / 16.3% | 19.8 / 70.2 |
| 11 | 16 | 1.73 / 0.08 / 0.08 | 57% | 0.339 / 0.864 | +3.14 / +2.00 | 35.9% / 21.9% | 17.5 / 75.0 |
| 12 | 17 | 1.52 / 0.08 / 0.07 | 55% | 0.489 / 0.898 | +2.32 / +1.86 | 32.8% / 17.9% | 14.8 / 68.0 |
| 13 | 16 | 0.21 / 0.04 / – | 41% | 0.569 / 0.890 | +1.00 / +1.81 | 32.2% / 20.9% | 11.0 / 64.0 |
| 14 | 16 | 0.44 / 0.03 / – | 45% | 0.490 / 0.862 | +1.91 / +1.61 | 38.3% / 22.3% | 14.1 / 71.7 |
| 15 | 16 | 0.28 / 0.04 / – | 42% | 0.463 / 0.861 | +1.30 / +1.68 | 36.5% / 19.8% | 15.4 / 56.7 |
| 16 | 16 | 0.22 / 0.03 / – | 41% | 0.395 / 0.856 | +1.46 / +1.56 | 39.3% / 21.4% | 17.5 / 62.5 |
| 17 | 16 | 0.23 / 0.03 / – | 40% | 0.428 / 0.867 | +1.26 / +1.50 | 36.2% / 19.1% | 13.4 / 59.7 |
| 18 | 17 | 2.28 / 0.03 / 0.07 | 60% | 0.431 / 0.857 | +4.68 / +1.73 | 42.3% / 19.5% | 25.4 / 64.4 |

**Were the first two episodes typical?** The first round ran episodes 1 and 2 before the rest. The rule was recorded in
the log before the full round: typical if, on the three main measures (AVC ~ HEVC Spearman, AVC ~ HEVC median logit
difference, AVC enrichment in 5 s windows) and for both models, both episodes lie inside the range of the other 16, and
U3 − Small keeps the first round's sign with an interval that excludes 0.

| measure | episodes 1 / 2, U3 | other 16, U3 | episodes 1 / 2, Small | other 16, Small | U3 − Small, first round → all 18 |
|---|---|---|---|---|---|
| Spearman | 0.328 / 0.290 | 0.339–0.569 | 0.828 / 0.778 | 0.823–0.902 | −0.494 → −0.433 [−0.463, −0.404] |
| median logit difference | +5.57 / +4.20 | +1.00 to +4.68 | +2.17 / +2.09 | +1.50 to +2.15 | +2.75 → +0.96 [+0.25, +1.64] |
| enrichment, 5 s | 24.2 / 23.0 | 10.7–25.4 | 69.4 / 81.8 | 56.7–81.9 | −52.0 → −53.7 [−57.5, −49.7] |

- **By the rule they were not typical.** The direction holds on all three measures. Enrichment is inside the range.
  On the other two, episode 1 or 2 falls outside it, in both models.
- **The first round overstated the difference in level and in ranking** (Update 3's logit gap +4.88 against +2.78 over
  the series; its Spearman 0.309 against 0.429).
- **It understated the decision changes,** which were not among the three measures. There Update 3 and Small looked
  close (25.1% against 22.3%). Over the series Update 3 changes more in all 18 episodes (29.7% against 20.6%), and in
  episodes 13–18 it changes 32–42% against 19–22%.

**What the full round shows:**

- **AVC against HEVC.** In all 18 episodes both models read the AVC file blurrier. Update 3's gap is larger in 14 of
  18 and varies far more (+1.00 to +5.57 logits; Small +1.50 to +2.17). Update 3 also ranks the frames of the two files
  much less alike (Spearman 0.429 against 0.862, lower in all 18).
- **HEVC against AV1: Update 3 is the steadier.** Spearman 0.752 against 0.650 (higher in 10 of 12), and 17.1% of the
  r95 decisions change against 27.0% (fewer in 11 of 12).
- **AVC against AV1:** Update 3 ranks less alike (0.450 against 0.622), with the same share of decision changes.
- **Enrichment.** On AVC Update 3's is lower in every episode and window: per episode 0.14–0.40 of Small's in 5 s
  windows (median 0.23), and still large. On HEVC it is higher than Small's in every episode at 1 and 5 s, though small
  in itself (4.5× against 1.8× at 5 s). On AV1 the two do not differ at 5 or 20 s.
- **The share called sharp at r95, by frame type** (median over the episodes):

  | encode | Update 3 | Small |
  |---|---|---|
  | AVC | I 25%, P 19%, B 23% | I 84%, P 22%, B 28% |
  | HEVC | I 33%, B 35% | I 37%, B 45% |
  | AV1 | I 35%, P 31% | I 47%, P 50% |

- **Update 3's px level.** Median px per file: AVC 2.16 (0.21–3.15), HEVC 0.11 (0.03–0.23), AV1 0.15 (0.07–0.48).
  Frames over 1 px: AVC 60% (40%–73%), HEVC 18% (9%–29%), AV1 23% (17%–40%).
  - By frame type on AVC, over the episodes: I 0.03–0.69, P 0.56–3.56, B 0.18–3.08 px. On HEVC: I
    0.03–0.71, B 0.03–0.23 px.
- **The px gap to HEVC depends on how it is taken.** As a gap of the two medians: 1.97 px (0.17–3.01). As a paired
  per-frame median: 0.90 px (0.03–2.32). On AVC the median moves a long way when the share of frames over 1 px
  crosses one half.

**Added: the same frame, another frame type** (`eval/yr_typepairs.py`). AVC − HEVC in logits, median over the episodes
(range):

| frame type in AVC / HEVC | frames | Update 3 | Small |
|---|---|---|---|
| I / I | 11,337 | +0.15 (−0.18 to +0.62) | −7.84 (−11.51 to −4.54) |
| B / B | 924,164 | +2.68 (+1.00 to +5.52) | +1.77 (+1.46 to +2.13) |
| P / B | 192,076 | +3.45 (+1.21 to +6.13) | +2.21 (+1.88 to +2.57) |

- Where a frame is an I-frame in both files, Update 3 reads them alike. Small reads the AVC I-frame far sharper, its old
  I-frame bias in x264. Where a frame is P or B, both read AVC blurrier, Update 3 by more.

**Added: noticed after the full round, not registered** (`eval/yr_file_facts.py`):

- **The same settings, a very different reading.** In episodes 13–17 Update 3 reads the AVC file at 0.21–0.44 px
  (40–45% of the frames over 1 px), and in episode 18 at 2.28 px (60%).
  - The HEVC files read alike in all six (0.03–0.04 px). The x264 settings differ only in the CRF (16 against 17).
  - Small's logit gap to HEVC hardly moves (+1.50 to +1.81 against +1.73); Update 3's jumps (+1.00 to +1.91 against
    +4.68).
  - Episodes 1 and 18 (CRF 17) have the two largest gaps of the series, but episode 12, also CRF 17, does not (+2.32).
    So the CRF alone does not explain it, and the cause is not established.
  - So Update 3's px compare only within one encode. The same codec and settings do not make two files comparable.
- **Episode 7's AV1 file, without film grain, behaves differently** (one episode, at 0.75 Mbit/s, not a test):
  - Small agrees with it far better than with the other AV1 files: Spearman with HEVC 0.913 against 0.577–0.669, r95
    changes 12.4% against 25.8–31.0%; Spearman with AVC 0.864 against 0.494–0.666.
  - Update 3's agreement with HEVC stays inside the others' range (0.801 against 0.633–0.841).
  - Update 3 reads it blurrier: 0.48 px against 0.07–0.21 px, 40% of the frames over 1 px against 17–26%; the median
    logit difference HEVC − AV1 is −1.21 against −0.68 to +0.50.
  - This suggests that Small's low agreement with the other AV1 files comes from the synthesised grain rather than the
    codec, and that grain lowers Update 3's reading. One file cannot show either.
- **Inside one file** the two models rank the frames alike on AVC (Spearman 0.858, 0.758–0.908) and much less
  alike on HEVC (0.424, 0.335–0.497) and AV1 (0.416, 0.330–0.588). On HEVC, across Small's deciles 1–9 Update 3
  reads 0.01–1.43 px; the top decile reads 1.70 px (0.34–3.35).

**The light-jump control.** In the source encode a flash starts a keyframe (recorded before the results). The values per
episode are in [data/yr.json](data/yr.json); episode 1, the episode chosen for its flash scenes:
- On episode 1 AVC, the share called sharp at r95, Update 3 against Small: an I-frame with a light jump 34% against
  93%; an I-frame without one 18% against 89%; a P-frame with a light jump 3% against 20%.
- The I-frame's logit against the median of its ±4 neighbours: Update 3 −3.41 (no jump) and −6.77 (jump), Small
  −14.0 and −16.8.

**Added: why the encodes differ.** Two explanations were checked on episode 1:
- **Dither:** the 10 → 8 bit conversion does not dither on this path. The decoded pixels are byte-identical with
  every `sws_dither` setting.
- **Grain in HEVC:** the HEVC frames show no extra grain. Their Laplacian RMS is lower than the AVC frames' (2.03–2.78
  against 2.24–3.08, three 1 s segments). This does not cover the AV1 file, which has synthesised grain.

Then one minute of episode 2 (1,440 frames) was decoded from HEVC and re-encoded:

| variant | rate | PSNR to the HEVC frames | U3 px | U3 Spearman with the HEVC frames | Small logit | Small Spearman |
|---|---|---|---|---|---|---|
| HEVC (reference) | 3.9 Mbit/s | | 0.62 | 1 | −0.10 | 1 |
| 8-bit 4:2:0 lossless (FFV1, x264 qp 0) | 102–105 Mbit/s | 44.6 dB | 0.66 | 0.875 | −0.03 | 0.976 |
| 10-bit 4:2:0 lossless (FFV1) | 194 Mbit/s | 42.5 dB | 0.67 | 0.867 | −0.22 | 0.976 |
| x264 CRF 12 | 19.5 Mbit/s | 43.0 dB | 1.20 | 0.733 | +2.19 | 0.877 |
| x264 CRF 18 | 8.4 Mbit/s | 41.7 dB | 2.18 | 0.558 | +2.63 | 0.842 |
| x264 2.7 Mbit/s | 2.5 Mbit/s | 39.4 dB | 3.04 | 0.193 | +3.36 | 0.714 |
| x265 10 bit 2.7 Mbit/s | 2.5 Mbit/s | 39.2 dB | 1.72 | 0.797 | +0.41 | 0.873 |
| the AVC file (stored scores) | 4.5 Mbit/s | | 3.26 | 0.255 | +3.82 | 0.782 |

(x264 preset slow, keyint 240, 3 B-frames; x265 preset medium, keyint 240. The rates are those of this minute.)

**How far this test goes:**
- It is one minute of one episode, and not a typical minute. The AVC and HEVC files spend 1.9 and 1.7 times their average
  rate on it, and Update 3 reads them at 3.26 and 0.62 px, against 2.77 and 0.19 px over the episode.
- The re-encodes are second generation: their input is the HEVC file decoded to 8-bit RGB. The AVC file is x264 at
  CRF 16 with other settings (16 B-frames, for example), from a source not available here. So equal CRF values do not
  compare: in this minute the file at CRF 16 takes 4.5 Mbit/s, and the re-encode at CRF 18 takes 8.4 Mbit/s.
- x264 ran in 8 bit and x265 in 10 bit, so this table does not separate the codec from the bit depth (see the 2 × 2 test
  below).
- Both 2.7 Mbit/s runs got less than the reference's 3.9 Mbit/s, so part of x265's rise may be the lower rate.

**What it shows within these limits:**
- **x264 coding comes close to the AVC file's reading.** The same pixels through x264 at 2.5 Mbit/s read 3.04 px, with
  Spearman 0.19. The AVC file, at 4.5 Mbit/s, reads 3.26 px, with 0.26.
- **The effect grows step by step as the quality falls,** and is there even at 43 dB (+0.6 px). The 4:2:0 / 8-bit
  conversion alone does nothing.
- **Update 1 Small reacts the same way,** by as much or more in logits (CRF 12: −0.10 → +2.19, across the decision
  boundary). It keeps its ranking better.
- **In this test both models read x264 coding of film frames as blur,** even at 43 dB, and x265 10 bit much less.
  - The effect spans the whole coding, not the frame type alone.
  - Every video of the training corpus is H.264 8 bit, so the frame labels were all learned on such frames.
- **What is Update 3's own:**
  - its px scale makes the effect visible (0.6 → 1.2–3.0 px);
  - its ranking is more sensitive to small changes (lossless round trip: 0.875 against 0.976). On the 10-bit encodes
    most frames fall into its unresolved band under 1 px, where the order is noisy.
- On the photo clips (item 3(b)) x264 CRF 23 left a 0 px photo at 0.01 px. On film the same coding adds 1–2 px. Why
  is open; the removal of grain is a suspicion, not a measurement.

**Added: codec against bit depth** (`eval/yr_depth.py`; the decision rule was recorded in the log before the run). The
same minute in a 2 × 2 design, with the encoder settings above.

| variant | rate | PSNR to the HEVC frames | U3 px | U3 logit | U3 Spearman with the HEVC frames | Small logit | Small Spearman |
|---|---|---|---|---|---|---|---|
| HEVC (reference) | 3.9 Mbit/s | | 0.62 | −0.31 | 1 | −0.10 | 1 |
| x264 8 bit | 2.5 Mbit/s | 39.39 dB | 3.06 | +4.75 | 0.188 | +3.46 | 0.717 |
| x264 10 bit | 2.5 Mbit/s | 38.65 dB | 2.49 | +3.64 | 0.289 | +4.12 | 0.703 |
| x265 8 bit | 2.5 Mbit/s | 40.23 dB | 1.96 | +2.62 | 0.669 | +1.69 | 0.817 |
| x265 10 bit | 2.5 Mbit/s | 39.24 dB | 1.73 | +2.16 | 0.790 | +0.44 | 0.874 |

The two repeated variants match the first pass (U3 3.06 against 3.04 px and 1.73 against 1.72 px; Small +3.46 against
+3.36 and +0.44 against +0.41).

The registered measures, as paired per-frame medians with a block bootstrap over 2 s blocks. The rule calls it "codec"
if |codec| ≥ 2 |depth| and the codec interval excludes 0, "depth" the other way round, and otherwise "not separated".

| | U3 logit | U3 px | Small logit |
|---|---|---|---|
| codec (x264 − x265) | +0.76 [−0.17, +1.51] | +0.25 [−0.00, +0.61] | +1.88 [+1.55, +2.39] |
| bit depth (8 − 10 bit) | +0.49 [+0.18, +0.82] | +0.21 [+0.06, +0.38] | +0.02 [−0.13, +0.19] |
| interaction | +0.13 [−0.37, +0.70] | +0.06 [−0.08, +0.32] | −1.01 [−1.22, −0.80] |
| **decision by the rule** | **not separated** | | **codec** |

Worked out after the decision, as a description ([data/yr_depth_simple.json](data/yr_depth_simple.json)):

| | U3 logit | U3 px | Small logit |
|---|---|---|---|
| 8 − 10 bit, in x264 | +0.51 [+0.16, +0.99] | +0.22 [+0.05, +0.41] | −0.50 [−0.65, −0.27] |
| 8 − 10 bit, in x265 | +0.38 [+0.25, +0.58] | +0.10 [+0.05, +0.22] | +0.47 [+0.34, +0.64] |
| x264 − x265, in 8 bit | +1.05 [−0.06, +1.88] | +0.34 [−0.01, +0.80] | +1.33 [+1.04, +1.74] |
| x264 − x265, in 10 bit | +0.54 [−0.23, +1.31] | +0.13 [−0.05, +0.45] | +2.35 [+1.99, +2.91] |
| x264 8 bit − reference | +3.29 [+1.26, +4.93] | +1.29 [+0.32, +2.36] | +2.28 [+1.77, +2.80] |
| x264 10 bit − reference | +1.88 [+0.79, +3.13] | +0.64 [+0.16, +1.36] | +2.75 [+2.08, +3.82] |
| x265 8 bit − reference | +1.31 [+0.94, +2.09] | +0.43 [+0.18, +0.96] | +0.63 [+0.33, +1.00] |
| x265 10 bit − reference | +0.94 [+0.66, +1.42] | +0.28 [+0.12, +0.60] | +0.15 [−0.13, +0.44] |

- **Update 3: the rule does not decide.** Both effects point the same way, x264 above x265 and 8 bit above 10 bit.
  - The bit-depth effect is small, but its interval excludes 0 in both codecs. It runs against PSNR: at the same rate the
    8-bit encodes are 0.7–1.0 dB better.
  - The codec effect has the larger point value, but it varies strongly inside the minute and its interval crosses 0.
- **Update 1 Small: the codec, by the rule.** Its near-zero bit-depth effect hides opposite signs (x264 −0.50, x265
  +0.47), so it does not show that bit depth has no effect. x265 10 bit does not differ from the reference.
- **Every re-encode raises Update 3's reading against the reference** (+0.94 to +3.29 logit), with point values above
  either effect.
- **The medians in the first table can mislead.** x264 and x265 in 8 bit differ by 1.1 px in their medians but by 0.34 px in
  the paired median, because the frames respond very unevenly.
- **Limits:**
  - one minute, not a typical one, and second-generation encodes;
  - the input is 8-bit RGB, so 10 bit here is the encoder's precision, not a 10-bit source as in the real files;
  - the source is HEVC, so x265 re-codes the pixels with tools related to the source's, which may favour it (not tested;
    noticed after the results).

  So this does not explain the difference between the AVC and the 10-bit files.

**Added: two H.264 releases of the same film** (`eval/two_releases.py`; the Update 1 README's consistency check).
- 31 corpus films, both releases H.264 8 bit, 1920 × 1080. Most of these films are in the training split.
- The lower release runs at 4.8–10.0 Mbit/s (median 6.6), the higher at 7.5–12.4 (median 9.3), audio included.
- 400 time-matched pairs per film, 12,400 in all.

| | Update 3 | Small | U3 − Small (median over films; bootstrap over films) | U3 higher |
|---|---|---|---|---|
| Spearman of the logits | 0.801 | 0.793 | +0.009 [+0.001, +0.014] | 21 / 31 |
| r95 agreement | 85.0% | 84.5% | +0.003 [−0.008, +0.012] | 17 / 31 |
| κ at r95 | 0.678 | 0.669 | +0.009 [−0.015, +0.030] | 18 / 31 |
| median logit, lower − higher rate | +0.055 | +0.037 | +0.016 [−0.068, +0.051] | 17 / 31 |

- **Between two good H.264 releases Update 3 is as stable as Small,** with no level difference (0.018 against
  0.015 px).
- Small reproduces the Update 1 Base corpus figure (85.0%, κ 0.68).
- **So the test series' difference is not a general instability between encodes.** In the 18 episodes tested it is the
  AVC encode against the 10-bit ones. These releases run well above the series' rate and are all H.264, so they do
  not test low-rate H.264 or other codecs.

## 7. Item 6: px on native frames

**By VLM code** (held-out films):

| code | n | median px | under 1 px | inner cells' p90, median |
|---|---|---|---|---|
| `sss` | 67,609 | 0.005 | 93.4% | 0.006 |
| `ssb` | 3,401 | 0.012 | 80.6% | 0.005 |
| `bsb` | 14,948 | 2.40 | 26.4% | 2.04 |
| `bbb` | 28,729 | 4.21 | **5.6%** | 4.90 |

- R4T-1 measured 17–29% of the `bbb` frames under 1 px; Update 3 measures 5.6% (5.2% on the validation frames).
- Order AUC with the frame px: `bbb` > `bsb` 0.819, `bsb` > `sss` 0.932, `bbb` > `sss` 0.981; with the inner cells'
  p90: 0.835 / 0.919 / 0.983. This order is partly met by construction (the label trains the score the px reads).

**Inside a code, against the measured motion** (`region_px`; Spearman; shot bootstrap):

| code (n; median motion) | U3 frame px | U3 cell p90 | U3 m_inner | Small logit | Small m_inner | Small max_inner |
|---|---|---|---|---|---|---|
| `sss` (67,609; 6.0 px) | 0.409 | 0.422 | 0.390 | 0.433 | 0.410 | 0.382 |
| `ssb` (3,401; 5.3 px) | 0.505 | 0.657 | 0.458 | 0.526 | 0.521 | 0.446 |
| `bsb` (14,948; 9.7 px) | 0.437 | 0.617 | 0.582 | 0.411 | 0.585 | 0.268 |
| `bbb` (28,729; 15.0 px) | 0.569 | 0.619 | 0.642 | 0.498 | 0.639 | 0.411 |

- Update 3's frame px follows the motion better than Small's frame logit (`bbb` +0.071 [+0.054, +0.089], `bsb` +0.026
  [+0.010, +0.044]; `sss` −0.024 [−0.045, −0.002]).
- The same cell readout (m_inner) follows it equally in both (`bbb` +0.003 [−0.013, +0.021], `bsb` −0.003 [−0.017,
  +0.011]; `ssb` −0.063 [−0.090, −0.035]). The frame-level gain is a difference of readout, not of the map.
- Update 3's cell p90 is better than Small's best cell readout on `ssb` and `bsb` (+0.137 [+0.073, +0.198], +0.033
  [+0.010, +0.054]) and level on `bbb` (−0.020 [−0.054, +0.018]).
- With item 4: inside a frame Update 3's cell map ranks motion worse; across frames, inside a code, it is not worse.

## 8. Not done

- The author's visual inspection of the cell maps (item 4). Update 3 is now in FrameBlurStudio, so it can be done.
- The 2×2 design's other runs, so the three changes stay confounded.
