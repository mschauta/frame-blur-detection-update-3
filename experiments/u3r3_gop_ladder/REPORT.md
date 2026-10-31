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
| 5. agreement of the AVC and the 10-bit encodes of the same frames (Spearman; 2 episodes) | 0.29–0.40 | 0.60–0.83 | – (worse) |
| 5. Phase II I-frame enrichment, AVC, 5 s windows (2 episodes) | 23–24× | 69–82× | – |
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
- **On film (two episodes of one series) the frame-type bias is reduced, not removed, and the encode still matters.**
  - On the AVC encodes of the test series the I-frame enrichment falls to a third or less of Small's (23–24×
    against 69–82× in 5 s windows).
  - But Update 3 reads the AVC encode 2.5–3 px blurrier than the 10-bit HEVC and AV1 encodes of the same
    frames, and ranks the frames of the two very differently (Spearman 0.29–0.40; Small 0.60–0.83).
  - In a one-minute re-encoding test, x264 coding of film frames raised the reading in steps with the quality, by
    0.6 px even at 43 dB, while x265 10 bit at the same rate raised it less. The test does not separate the codec
    from the bit depth, and its encodes are second generation (section 6).
  - Update 1 Small reacts the same way in logits, and its decision crosses the boundary; it keeps its ranking
    better.
  - On two H.264 releases of 31 corpus films, Update 3 agrees as well as Small (Spearman 0.801 against 0.793).
- **The cell map localises motion less well** inside a frame than Update 1 Small's (0.249 against 0.338).
- **The 95% point is very strict.** Its threshold sits at 0.05 px, so it calls 21% of coded 0 px photos blurred
  (Update 1 Small: 4%), as strict as on the sharp validation frames.

One run, three changes (photo coding, continuous ladder, no masks): no result here can be assigned to one of them.

## Deviations from the protocol

- **Item 5 covers a subset.** The protocol registers 18 episodes in AVC and HEVC and 12 in AV1. Episodes 1 and 2 were
  run in AVC and HEVC, and episode 2 in AV1.
  - The episodes were chosen on 2026-10-30, before the item 5 results: episode 1 for its flash scenes, episode 2 as the
    first in all three codecs.
  - The full round would take about 21 GPU hours at the measured speed, and it was not run.
  - The log and the docstring of `eval/yr.py` wrongly call this a choice within the protocol. The protocol fixes the
    episodes too.
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

Every frame of episode 1 in AVC and HEVC, and of episode 2 in AVC, HEVC and AV1, of the test series (not named),
scored with both models; the whole 1920 × 1080 frame with its bars, as FrameBlurStudio scores. The files of one
episode align frame by frame (the luma sequences correlate at 0.9999 at zero offset).

| encode | codec, bit depth | video rate (with audio) | frames |
|---|---|---|---|
| episode 1 AVC | H.264 High, 8 bit | 2.20 Mbit/s (2.59) | 62,674 |
| episode 1 HEVC | HEVC Main 10 (I and B frames) | 2.48 Mbit/s (3.13) | 62,674 |
| episode 2 AVC | H.264 High, 8 bit | 2.34 Mbit/s (2.73) | 72,968 |
| episode 2 HEVC | HEVC Main 10 | 2.30 Mbit/s (2.75) | 72,968 |
| episode 2 AV1 | AV1 Main, 10 bit, film grain | 1.49 Mbit/s (1.94) | 72,968 |

- The encoders of these files are not known: AVC and HEVC name the format here, not x264 or x265.
- **The AV1 file carries film-grain synthesis.** `apply_grain` is set in 68,246 of its 72,968 frame headers, and the
  decoder used here applies it. So its readings are of grained frames, and it cannot show that the bit rate does not
  matter.

**Phase II I-frame enrichment** (the sharpest frame of each 1 / 5 / 20 s window, ties shared, dark frames excluded,
against the I-frame share of the windows; 95% interval for 5 s):

| | Update 3 | Small |
|---|---|---|
| episode 1 AVC | 9.9 / 24.2 [20.9, 27.6] / 17.1 | 21.0 / 69.4 [65.1, 73.7] / 82.7 |
| episode 1 HEVC | 2.5 / 4.5 [3.0, 6.0] / 7.5 | 1.6 / 1.4 [0.5, 2.4] / 1.2 |
| episode 2 AVC | 9.7 / 23.0 [19.4, 26.6] / 17.7 | 21.3 / 81.8 [77.8, 86.0] / 107.7 |
| episode 2 HEVC | 2.9 / 4.0 [2.4, 5.7] / 6.6 | 1.4 / 2.1 [1.0, 3.4] / 1.5 |
| episode 2 AV1 | 2.8 / 3.9 [2.3, 5.8] / 3.1 | 1.6 / 2.8 [1.4, 4.3] / 3.1 |

- On AVC Update 3's enrichment is a third to a sixth of Small's, but still large. On HEVC and AV1 Small has almost
  none and Update 3 a little more.
- Update 3's median px by frame type: episode 1 AVC I 0.41, P 3.56, B 3.08; HEVC I 0.25, B 0.15. Episode 2 AVC I
  0.42, P 3.15, B 2.70; HEVC I 0.36, B 0.19; AV1 I 0.14, P 0.22.
- **The light-jump control.** In the source encode a flash starts a keyframe (recorded before the results).
  - On episode 1 AVC, the share called sharp at r95, Update 3 against Small: an I-frame with a light jump 34% against
    93%; an I-frame without one 18% against 89%; a P-frame with a light jump 3% against 20%.
  - The I-frame's logit against the median of its ±4 neighbours: Update 3 −3.41 (no jump) and −6.77 (jump), Small
    −14.0 and −16.8.

**Agreement between the encodes:**

| pair | U3 Spearman | Small Spearman | U3 r95 decision changes | Small r95 decision changes |
|---|---|---|---|---|
| episode 1 AVC ~ HEVC | 0.328 | 0.828 | 25.5% | 20.6% |
| episode 2 AVC ~ HEVC | 0.290 | 0.778 | 24.6% | 24.0% |
| episode 2 AVC ~ AV1 | 0.395 | 0.603 | 25.1% | 26.8% |
| episode 2 HEVC ~ AV1 | 0.633 | 0.664 | 19.9% | 25.9% |

- **Level:** Update 3 reads the AVC encodes at 2.77–3.15 px (median) and the HEVC and AV1 encodes at 0.15–0.21 px.
  Small reads AVC about 2 logits blurrier than HEVC.
- **Added: the same frame, another frame type** (`eval/yr_typepairs.py`):
  - where a frame is an I-frame in both encodes, Update 3 reads AVC and HEVC alike (+0.29 and +0.03 logit). Small
    gives −11.5 and −10.6, its old I-frame bias in x264;
  - where it is P or B, Update 3 reads AVC 4.1–6.1 logits blurrier (Small 2.0–2.5).
- On the 10-bit encodes, across Small's deciles 1–9 Update 3 reads 0.05–0.52 px; only the top decile reaches 1.2–2.2
  px. On AVC Update 3 follows Small's order (Spearman 0.87–0.91).

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
- The re-encodes are second generation: their input is the HEVC file decoded to 8-bit RGB. The AVC file was coded once,
  by an unknown encoder.
- x264 ran in 8 bit and x265 in 10 bit, so the codec and the bit depth are not separated.
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
- **So the test series' difference is not a general instability between encodes.** In the two episodes tested it is the
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
- The other 16 episodes in AVC and HEVC and the other 11 in AV1 (item 5).
- The 2×2 design's other runs, so the three changes stay confounded.
