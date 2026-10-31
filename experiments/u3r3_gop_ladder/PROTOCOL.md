# U3-R3: a continuous, GOP-coded photo ladder on the round3 recipe (ConvNeXt-Small)

English translation of the registered protocol, [PROTOCOL_hu.md](PROTOCOL_hu.md). The Hungarian text was fixed on
2026-10-29, before the run started. Where the two differ, the Hungarian text governs.

**The name of the test series.** The registered text names a TV series used as test material in two places (evaluation
items 4 and 5). Until 2026-11-01 the copies here replaced the name with a placeholder. Both now carry the registered
wording, so `PROTOCOL_hu.md` matches the registered SHA-256 in [PROTOCOL.sha256](PROTOCOL.sha256).

**Names.** File names refer to the training code as run; that code is not published. "round3" is the internal name of
the Update 1 recipe: round3 Base is Update 1 (release v1.0), round3 Small is Update 1 Small (release v1.1), and v9 is
Update 2. FrameBlurStudio is the author's frame-selection application.

## Protocol (fixed on 2026-10-29, before the run started)

**Question** (the author, 2026-10-29): what happens if, in the round3 recipe, the photos get a continuous 0–30 px
ladder, and the images are coded as GOP clips in all three codecs ("in effect we would refine Update 1")?
1. Does the codec dependence decrease (R4K-2)?
2. Does the model rank inside the sharp band (0–3 px)?
3. Is the frame accuracy kept?
4. What cell map does it give?

**Decisions** (the author, 2026-10-29):
- Only this run (cell 3 of the proposed 2×2 design).
- All three codecs, so that it can be tested on every video in FrameBlurStudio.
- No watermark mask, and the mask leak need not be handled.
- The ordinal output with a floating anchor, as proposed (below).

**Run:** `runs/update3_small_gop_ladder_v1`, configuration `configs/update3_small_gop_ladder.json`, launcher
`round3_inditas.bat configs\update3_small_gop_ladder.json`.

### The recipe

**What differs from round3 Small:**
1. **Photos:** a continuous, GOP-coded ladder (below), instead of the two binary photo classes (0–3 px sharp,
   16–30 px blurred).
2. **Head:** an ordinal output with a floating anchor, instead of the binary photo target.
3. **Mask:**
   - no watermark mask on the frames or on the photos (`frames_mask_share` and `synth_mask_share` 0.5 → 0);
   - `mask_leak_fix` off.
   The frames therefore learn together with the watermark pixels, as in v9 and as FrameBlurStudio scores.
4. **Data loader:** 16 processes instead of 10, because of the costlier synthesis.

**What is the same:**
- the index (`data/round3_index.csv`), the frame selection (v8), the frame labels: sharp = sss (+ ssb in the
  ssb-dominant films), blurred = bbb + bsb;
- the strata: frames/sharp 12,000, frames/blurred 12,000, photo_ladder/ladder 24,000 per epoch. The photo : frame
  ratio is 1 : 1, as in round3;
- 30 epochs × 48,000 samples, batch 8 × 2;
- the learning rate (1e-4 / 1e-3), the warm-up (0.25 epoch), the cosine decay to 1%, the seed (20261016);
- the fp32 fingerprint, ConvNeXt-Small from the ImageNet-1K weights, the LSE pooling (r = 4 at the start);
- the weight of the dense (cell-level) photo loss 0.5, the edge threshold 0.001.

**The photo sample** (`synth.gop_recipe` and `make_gop`, deterministic from the (photo, epoch) pair):
- **Length:** 20% exactly 0 px, 40% uniform in (0, 8] px, 40% uniform in (8, 30] px.
- **Direction:** any (0–180°). Camera motion in linear light (`motion_kernel_v2`).
- **Clip:** 8 frames, 30 fps. The crop loses 16 px per side. Panning along the blur direction, 0–4 whole px per
  frame, with equal probability (0 is the static clip). The clip consists of windows cut from the blurred photo; there
  is no resampling.
- **Codec:** one third each:
  - x264 (high, preset medium, 8 bit);
  - x265 (preset fast, 10 bit);
  - AV1 (libaom, cpu-used 8, 10 bit).
  - 30% of the AV1 samples get film-grain synthesis (denoise-noise-level 5–25).
- **Structure:** I only 0.2, IP 0.3, IPB 0.5.
  - x264 and x265: keyint 8, 3 B-frames, closed GOP, scenecut 0.
  - AV1: IPB = lag-in-frames 8 (hidden alt-ref frames), IP = lag 0.
- **CRF:** from the codec's list. The lists are matched by PSNR to about 35–41 dB (on 24 training photos,
  2026-10-29):
  - x264: 18, 21, 23, 26, 29, 32;
  - x265: 20, 23, 26, 29, 32, 34;
  - AV1: 24, 32, 38, 44, 50, 55.
- **Chain:** bt709, tv range, 4:2:0.
- **The sample** is one frame of the clip, with equal probability; every frame carries the same L. For the cell
  target the structure is taken from the same window of the original (unblurred) photo.
- **Validation:** a fixed variant of the validation photos (epoch None).

**The head:**
- Thresholds: T = 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 30 px.
- logit(L ≥ t_k) = s − b_k. b is monotone (cumulative softplus gaps plus a learned offset), and no threshold is tied
  to 0.
- The frames' binary VLM label trains BCE(s) at image level, so s = 0 is the frames' own boundary. The cell and image
  logits therefore stay the logits of the frame decision: the thresholds, the export and FrameBlurStudio work
  unchanged.
- A photo gets a target for every threshold: at image level, and per cell on every structured cell.
- At the start the 3 px threshold sits at 0.
- Measured length: L̂ = Σ σ(s − b_k)(t_k − t_{k−1}).

**Chosen epoch:** fixed in advance: the last, epoch 29 (as for round3 Base and Small). best.pt is not used.

**Preliminary checks:**
- 40 unit tests (`tests/test_gop.py` and the earlier ones);
- a trial run with a small index, 2 epochs;
- a stress test: 1,200 clips, 0 failures;
- in the trial run one ffmpeg crash (0xC0000005), handled by the retry.

### Control

- round3 Small, epoch 29 (`round3_small_p99_ep029.pt`).
- In R4K-2 also round3 Base and v9; there the published numbers are the basis of the comparison.

### Evaluation (with the epoch 29 weights)

Export: in Update 1's format. The thresholds (r90 / r95 / r98) come from the validation frames, as before. The meta
also holds the ordinal thresholds (for the px estimate).

1. **Codec dependence, R4K-2 (fixed test, unchanged clips and analysis).**
   - The new model on the existing 115,200 frames.
   - Primary measure: the paired logit difference between P/B and I-only, and the difference in the share called
     blurred at r95, per rate and per motion.
   - Comparison with round3 Small: the difference of the codec gap (new − Small), with the same photo resamples.
2. **Held-out films (the procedure of R4S-1).**
   - Measures: AUC, sharp frames kept at r90 / r95 / r98, the Spearman correlation with the motion measurement.
   - Paired difference to round3 Small (bootstrap over the shots).
   - The intervals refer to the shots of the six films.
3. **Photo ladder.**
   - (a) The published Update 1 ladder (300 photos, single-frame CRF 23): the flip point, the 0–3 px pair order,
     strict increase over 0–1–2–3 px.
   - (b) The same in GOP, x264 / x265 / AV1, with I-only and IPB structure only: the deviation of the measured px
     from the true L (MAE, Spearman) per codec and per frame type.
4. **Cell maps.**
   - The author's visual inspection on known films (e.g. Young Royals S01E01).
   - The cell-level motion-measurement witness on the validation frames (the method of R4T-1 9.6).
5. **Young Royals codec benchmark.**
   - The same frame in the AVC and HEVC SDR versions (18 episodes), and in AV1 SDR (12 episodes): agreement of the
     scores, decision changes, the I-frame enrichment of Phase II.
   - Between the new model and round3 Small. In FrameBlurStudio or by script.
6. **px on the native frame.** The order by the VLM (Qwen) codes (bbb > bsb > sss) and the motion. R4T-1 failed
   there (17–29% of the bbb frames under 1 px).

### Interpretation (in advance)

- **Proposed criteria; the decision is the author's:**
  - codec dependence: the R4K-2 codec gap shrinks at least by half relative to round3 Small, and its interval does
    not contain 0;
  - frame accuracy: the interval of the held-out AUC loss does not cross −0.005;
  - ranking in the sharp band: the 0–3 px pair order is greater than round3 Small's 0.659.
- A single run: if it improves, the coding and the continuous ladder cannot be separated as causes. That would need
  the other two runs of the 2×2 design.
- Dropping the mask is a third difference from round3 Small. FrameBlurStudio scores without a mask, so the comparison
  happens on the same input.

### Not included

- The other runs of the 2×2 design; the fine-tune from round3 Small.
- HDR, and separate measurements of letterbox bars and dark scenes (`update3_terv.md` section 6).

## Log

The registered text ends here. The log entries follow it in the lab notebook and will be published with the report.
