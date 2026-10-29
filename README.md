# Frame Blur Detection — Update 3 (in training)

The fourth model of [Frame Blur Detection](https://github.com/mschauta/frame-blur-detection), after
[Update 1](https://github.com/mschauta/frame-blur-detection-update-1) and
[Update 2](https://github.com/mschauta/frame-blur-detection-update-2). The input is unchanged: the signed edge
fingerprint of the [RGB Mesh Resampling](https://github.com/mschauta/rgb-mesh-resampling) `m09` reconstruction. What
changes is what the photo anchors teach. The earlier models learned a class from them: sharp or blurred. Update 3
learns a length: how many pixels of camera blur a frame carries, from 0 to 30 px.

**Status, 2026-10-29: training.** There are no weights and no results yet. This repository records three things
before the results exist:

- how the three recipes differ;
- what the decision threshold means once the photos teach a length;
- the evaluation, fixed in advance: [experiments/u3r3_gop_ladder/](experiments/u3r3_gop_ladder/).

## The system

[SYSTEM.md](SYSTEM.md) is the full description of the system. For every part it says what the part was used for and
what it did, measured on the models of the series, and how to set it:

- **The generations:** Generation 1 learns from VLM-labelled frames and the photo ladder; Generation 2 learns from
  Generation 1's cell maps of every frame and the photo ladder. It covers what a generation inherits, what it gets
  anew, how the dependence on the VLM changes, and whether the student has surpassed its teacher so far.
- **Binary, length or blur mass:** whether blur can be a class at all, what each kind of output can say, where each
  fails, and which separates most cleanly.
- **The parts:** the dataset of native frames; the VLM teacher; the non-learned motion measurement; the frame
  selection; the photo ladder as etalon and regulariser, including what each step of the coding chain simulates;
  the signed edge fingerprint; the detector.
- **Training:** repetition against every-sample-new; the learning rate, schedule and noise; what is learned fast
  and what slowly; the choice of checkpoint; Small against Base.
- **Readouts and thresholds per model,** what each model measures and how well, the distillation, and the witnesses.
- **What generalises and what does not.**
- **A step-by-step recipe** for building an equivalent dataset and model.

## Development history, continued

Sections 1–8 of the history are in Update 1's README, section 9 in Update 2's. In short:

- A frozen vision-language model (VLM) labelled about one million video frames.
- ConvNeXt detectors learned the decision from the edge fingerprint, with one logit per 16 px cell.
- Synthetic photos with a known amount of added camera blur anchored the scale.

### 10. Update 3: from a class to a length

#### Update 1: a binary decision, a two-peaked photo ladder

- **Frames:** 589,663 selected frames, each with its binary VLM label.
- **Photos:** two classes with a gap between them:
  - sharp: 0, 1, 2 or 3 px of added camera blur;
  - blurred: 16, 20, 24 or 30 px.

  The gap was deliberate. With no transition between the two classes, the separation is easy to learn, and the
  photos agree with the binary frame labels.
- **The threshold:** the network gives a score, and the operating points (r90 / r95 / r98) are read off the
  validation frames.

#### Update 2: the predecessor's maps, a stricter ladder

- **The selection:** Update 1 scored the 5.01-million-frame corpus and gave every frame a cell map.
- **Frames:** Update 2 learned those maps on every frame of the training videos (4,850,458 frames), as soft targets
  per cell. No VLM label was used in training.
- **Photos:** the ladder became stricter on the blurred side, which now starts at 10 px instead of 16 px. The gap
  narrowed to 3–10 px, so moderate blur is pushed towards "blurred".
- **The frame value:** a fixed blur mass, with the threshold chosen when filtering.
- **Inherited bias:** because the teacher was Update 1, its errors were part of the target. Its frame-type bias is
  one of them (R4K-1, R4K-2).

#### Update 3: a continuous ladder, a length scale

- **Photos:** every length from 0 to 30 px occurs:
  - 20% exactly 0 px;
  - 40% in (0, 8] px;
  - 40% in (8, 30] px.

  A photo has no class, only its length. The model is no longer told where sharp ends.
- **Photo coding:** each photo goes through a short coded clip (x264, x265 or AV1; I-only, IP or IPB), and the
  sample is one decoded frame of it. In Update 1 and Update 2 a photo was never coded with inter frames.
- **Frames:** the frames and their binary VLM labels are Update 1's.
- **The start:** ConvNeXt-Small from ImageNet weights, 30 epochs, the Update 1 schedule. It is not a fine-tune of an
  earlier model.

| | Update 1 | Update 2 | Update 3 |
|---|---|---|---|
| frame supervision | binary VLM label | Update 1's cell map, soft per cell | binary VLM label (as Update 1) |
| photo ladder | sharp 0–3 px, blurred 16–30 px | sharp 0–3 px, blurred 10–30 px | continuous 0–30 px, no class |
| photo coding | single image | single image | 8-frame clip: x264 / x265 / AV1, I / IP / IPB |
| photo : frame samples | 1 : 1 | 1 : 3 | 1 : 1 |
| output | decision logit | blur mass M | decision logit plus a px scale |
| trunk | ConvNeXt-Base (v1.0), ConvNeXt-Small (v1.1) | ConvNeXt-Base | ConvNeXt-Small |

## What the threshold means in Update 3

Update 3 learns from two kinds of signal, and each sets a different part of the output.

- **The frames set the decision boundary.** The binary VLM label trains the frame score s, so s = 0 is the frames'
  own boundary between sharp and blurred: the VLM's notion of sharp. The cell and frame logits stay decision logits.
  The operating points r90 / r95 / r98 come from the validation frames as before, and the export follows Update 1's
  format.
- **The photos set the length scale.** The output has 16 ordinal thresholds (0.5, 1, 2, 3, … 24, 30 px):
  logit(L ≥ t_k) = s − b_k. The offsets b_k are learned from the photos only. No threshold is tied to 0.
- **The VLM boundary therefore gets a position in pixels,** and that position is learned. It started at 3 px. In
  the first three epochs it sat at 3.21, 3.11 and 2.84 px (interim values; the fixed checkpoint is epoch 29).

The class is then chosen when filtering, not in training, in line with the series' principle that 0 px is absolutely
sharp:

- **keep the VLM boundary,** about 3 px so far, which matches the earlier models' decision;
- **or set a stricter length,** for example 1 px, where the use needs it.

**The condition.** A px threshold on a real frame means something only if the length scale learned on photos
transfers to film frames. That is not shown yet. An earlier photo-only test (R4T-1, not yet published) learned the
scale on photos, but on native frames it measured 17–29% of the frames that the VLM called clearly blurred (bbb) as
under 1 px. Evaluation item 6 tests exactly this. Until it passes, a px value on a frame is a hypothesis.

## Why the photos must carry the codec correction

R4K-1 and R4K-2 showed that the coded frame type changes the scores. A sharp photo coded as a P- or B-frame scores
blurrier than the same photo coded as an I-frame. The question was where the bias comes from.

**The VLM labels do not carry it** ([experiments/u3_precheck_frametype/](experiments/u3_precheck_frametype/)). On
all 173 corpus videos a keyframe's label is barely sharper than its neighbours': 7.1% sharper than both, against
6.0% for non-I centres. Update 1 Base, on the same frames, gives the keyframe the lowest blur probability in 49.1% of
the cases, against 26.5%. The bias comes from the model, most likely from its photos, which were never coded with
inter frames: a sharp anchor always looked like an I-frame.

**The frames cannot remove it.** Neutral labels do penalise a model that scores an I-frame sharper than its
neighbours, but only weakly:

- about 1.2% of the labelled frames are keyframes, roughly 290 of the 24,000 frames per epoch if the training frames
  carry the corpus rate;
- the VLM labels are noisy.

**So the exact photo signal has to do it.** In Update 3 about 70% of the photo samples are inter (P or B) frames of
their clip. About 3,400 per epoch are inter frames of an exactly 0 px photo, against none in Update 1.

**The photo : frame ratio is kept at 1 : 1 deliberately,** as in Update 1, so that only the content of the photo side
changes. The ratio is the next variable if the correction does not transfer to film. Raising it has a cost: R4T-1
shows that photos alone do not carry the measurement to native frames, so a larger photo share can harm the frame
decision (evaluation item 2).

**Interim, validation photos:** by epoch 2 the I versus P/B gap has disappeared on the validation photos. At 0 px the
median measured length is 0.40 px on I targets and 0.36 px on P/B targets, and in no codec do the P/B targets measure
blurrier. This is the training distribution, so it was expected. It does not answer the question on film.

## The run

The full recipe is in the [protocol](experiments/u3r3_gop_ladder/PROTOCOL.md). In short:

- **Photos:** the continuous length (above), any direction, camera motion in linear light.
  - **Clip:** 8 frames with whole-pixel panning of 0–4 px per frame along the blur direction, cut from the blurred
    photo without resampling. Every frame carries the same length.
  - **Codecs:** x264 (8 bit), x265 (10 bit) and libaom AV1 (10 bit, film grain on 30% of the samples), one third
    each.
  - **Structure:** I-only 20%, IP 30%, IPB 50%.
  - **Quality:** CRF lists matched to about 35–41 dB PSNR across the codecs.
- **Frames:** Update 1's selection and labels, 24,000 per epoch. There is no synthetic blur on frames.
- **No validity masks.** Frames learn with the watermark pixels visible, as Update 2 did and as the scoring tools
  score.
- **Chosen checkpoint:** epoch 29, fixed in advance. The control is Update 1 Small (release v1.1, epoch 29), trained
  on the same frames from the same start.

## Evaluation, fixed before the results

The [protocol](experiments/u3r3_gop_ladder/PROTOCOL.md) was registered before the run started. Its SHA-256 is in
[PROTOCOL.sha256](experiments/u3r3_gop_ladder/PROTOCOL.sha256).

1. **Codec dependence:** the R4K-2 clips, unchanged, against Update 1 Small.
2. **Held-out films:** AUC and the sharp frames kept at the operating points, paired with Update 1 Small.
3. **Photo ladder:** Update 1's published ladder, and the same in GOP clips per codec and frame type.
4. **Cell maps:** visual inspection, and the cell-level motion witness.
5. **One series in three codecs** (AVC, HEVC, AV1): the same frames, agreement of the scores, and the I-frame
   enrichment when the sharpest frame of a time window is chosen.
6. **The px scale on native frames:** the order of the VLM codes and the motion.

The proposed criteria are in the protocol; the decision is the author's. Three caveats, recorded before the results:

- **Criterion 1 is partly met by construction.** The R4K-2 photos were never trained on, but the R4K-2 clips are the
  kind of clip Update 3 now trains on. The informative test of the hypothesis is the transfer to film: items 5
  and 6.
- **One run, three changes:** the coding, the continuous ladder and the dropped masks. If the result improves, the
  causes cannot be separated without the other runs of the 2×2 design (photo coding × ladder shape).
- **Six held-out films:** the intervals describe the uncertainty over their shots, not over other films.

## Data availability

As in the earlier repositories, the training data is **not published**: it was built from copyrighted videos and
photographs. No images, frames, crops or derived images from it are published. Published are:

- the method;
- the protocol;
- the aggregate evidence.

The test series is not named.

## Related work by the author

- [Frame Blur Detection](https://github.com/mschauta/frame-blur-detection): the method, the RGB / grayscale / edge
  fingerprint comparison and the first published weights.
- [Frame Blur Detection — Update 1](https://github.com/mschauta/frame-blur-detection-update-1): the larger corpus,
  ConvNeXt-Base and -Small, the float32 fingerprint. Update 3 keeps its frames and its recipe.
- [Frame Blur Detection — Update 2](https://github.com/mschauta/frame-blur-detection-update-2): learning from Update
  1's cell maps, the blur mass, and the two codec experiments R4K-1 and R4K-2.
- [RGB Mesh Resampling](https://github.com/mschauta/rgb-mesh-resampling): the continuous RGB-mesh reconstruction from
  which the edge fingerprint is derived.

## License

[MIT](LICENSE) © Dr. Schauta Marcell
