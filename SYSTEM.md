# The system behind the series

Every model of the series is built from the same four parts:

1. **the dataset**, the native video frames, which the work exists for;
2. **the regularisation**, a photo ladder with a known blur length, present throughout training;
3. **the distillation**, which uses the learned blur signal to label the dataset;
4. **optionally, a new model** trained on the distilled labels.

Each update is a setting of these four parts. The table at the end shows the settings side by side.

## 1. The dataset: native frames

**What it is.** 5,010,602 frames of 173 videos (H.264, 1920 × 1080, 4:2:0), held in a private dataset database.
- Every frame was decoded at the video's own frame rate and stored losslessly as PNG.
- A stored frame is therefore exactly the decoded picture. The coding artefacts of the source belong to it, and
  nothing is added.
- Frames with burnt-in text are excluded. A per-video mask marks the watermark.

**The blur is native.** It comes from real camera and subject motion and from defocus, and the types are not
separated. This is the blur that a frame selection has to judge.

**Two things are unknown for every frame:**
- **The blur length.** The exposure time is not known, so the non-learned motion measurement gives the displacement
  between neighbouring frames, not the length of the smear.
- **The blur map.** No frame says where in the picture it is blurred. There is no pixel- or cell-level truth.

**What is known about a frame:**
- **A VLM label** on 1,019,201 frames: three yes / no questions to a frozen Qwen3.5-4B (codes such as `sss`, `ssb`,
  `bsb`, `bbb`). It is a teacher's judgement and noisy. It does not depend on the coded frame type
  ([pre-check](experiments/u3_precheck_frametype/)).
- **The non-learned motion measurement:** camera and regional motion, and the sharpness rank inside a shot. It is a
  witness and a ranking tool, never a label and never a model input.
- **The structure:** shots, films, and two releases of 31 films, so the same picture exists at two bitrates.
- **The coded frame type** of the source (keyframe or not), read from the packets of the source file.

**Held out:** six films (3 validation, 3 test; 160,142 frames), in every update.

**Not published:** the frames are copyrighted material.

## 2. The regularisation: the photo ladder

**What it is.** Photographs, 48,797 originals with their own train / validation / test split, unchanged since the
first publication. Each original counts as sharp by construction: L = 0 px means no added blur. It is a sharp
composition, not a guarantee that every region is in focus. A training sample is built in four steps:

1. **Synthetic blur:** uniform camera motion of a known length L and direction, applied in linear light.
2. **Coding:** the blurred photo goes through the video path, coded and decoded, so that it reaches the model the way
   a frame does.
3. **Target:** from the known L, for the image and per cell. In Update 1 and Update 2 it was a class (sharp or
   blurred); in Update 3 it is the length itself.
4. **Augmentation:** flips and quarter turns, exact on the cell grid.

**Its role in training.** The photo ladder is the only exact signal in the system: the length is known, and it holds
everywhere in the picture. It is present in every epoch, from the first step to the last, beside the frames.
- **It anchors the scale** while the noisy frame labels pull the decision.
- **It gives the cell maps a dense target,** which the frames cannot give.
- **It is a regulariser, not the task.** A model trained on photos alone does not judge frames. In R4T-1, a
  photo-only length model measured 17–29% of the frames the VLM called clearly blurred (bbb) as under 1 px, and its
  cell maps did not follow the motion.

**What each update taught with it:**
- **Update 1:** two ends of the scale, 0–3 px sharp and 16–30 px blurred, with nothing in between. The transition is
  learned from the frames.
- **Update 2:** the blurred end from 10 px. By about epoch 10 the photo term was almost always met (median loss
  0.00017 against 0.071 for frames), so it held the ends while the frame targets set the transition. The photo
  ladder flips at 6.9 px, inside the untrained gap.
- **Update 3:** the whole scale, 0–30 px continuously, as a length, without classes.

**The coding path and its artefacts.** Whatever the photo path does to the picture, the model can learn as a
property of sharpness. Each such artefact has to be either matched to the frames or kept off the photos.
- **Uncoded photos would differ from the frames.** Synthetic blur turns photo grain into fine streaks, which the
  coding suppresses. This is why every photo is coded: H.264 at CRF 20–28, with CRF 23 closest to the frames on the
  residual statistics.
- **A photo coded alone is an I-frame.** In Update 1 and Update 2 every sharp anchor was intra-coded, while most
  video frames are P- or B-frames.
  - The models learned that sharp looks like an I-frame. R4K-1 observed it on two videos; R4K-2 confirmed it on
    sharp photos, for every model.
  - The VLM labels do not carry this bias, so it comes from the photo path.
- **Update 3 codes each photo inside a clip:** 8 frames, x264 / x265 / AV1, I-only / IP / IPB, whole-pixel pans,
  film grain on part of the AV1 samples. About 70% of the photo samples are inter frames.
- **Watermark masks.** In Update 1 and Update 2, half of the photo samples carried a watermark mask taken from the
  frames, so that the mask could not mark a photo. Update 3 uses no masks.
- **What the ladder does not cover:** subject motion, defocus, non-uniform blur. Photos teach uniform camera blur
  only.

**The photo : frame ratio.** It sets how much of each training step is exact signal.

| | photo samples per epoch | frame samples per epoch | ratio |
|---|---|---|---|
| Update 1 | 24,000 | 24,000 | 1 : 1 |
| Update 2 | 80,841 | 242,523 | 1 : 3 |
| Update 3 | 24,000 | 24,000 | 1 : 1 |

The ratio's own effect has not been measured. Update 2 changed it together with the frame targets and the ladder.
Update 3 keeps Update 1's ratio, so that only the content of the photos changes. Two observations bound it:
- **Too few photos, or the wrong photos:** the photo side defines what sharp texture looks like. With I-frame-only
  anchors every model became codec-dependent.
- **Too many photos:** photos alone do not carry the decision to native frames (R4T-1). A larger photo share can
  therefore cost frame accuracy.

The ratio is the next variable to test if Update 3's codec correction does not transfer to film.

## 3. The distillation: labelling the dataset with the learned signal

**The chain so far:**
1. The VLM labels about one million frames.
2. A detector learns the decision from them (Update 1).
3. The detector scores all 5,010,602 frames and gives each a cell map: one logit per 16 px cell, stored as an 8-bit
   integer at 1/4-logit resolution.
4. A new model learns those maps (Update 2): soft targets per cell, starting again from ImageNet.

**What the distillation adds:**
- **Coverage:** every frame, not only the labelled fifth.
- **Space:** a cell map instead of one label per frame.
- **Speed:** one network pass instead of three VLM queries.

**What it costs.** The student learns the teacher's errors, and its agreement with the teacher is not accuracy.
Update 1's frame-type bias was part of Update 2's target.

**The counterweights:** witnesses that the teacher did not produce.
- **The motion measurement:** in Update 2, a block with strong measured motion but a sharp teacher map was not
  taught.
- **The photo ladder:** an exact length.
- **The VLM labels of the held-out films.**
- **The two releases of the same film:** the same picture at two bitrates.
- **New in Update 3, the same frames in three codecs:** one series in AVC, HEVC and AV1.

**What Update 3 can add, if its length scale transfers to frames:** a length in px per frame and per cell. Three uses
would follow:
- ranking inside the block that the blur mass cannot order (M = 0; 29% of the held-out frames under Update 2);
- choosing the sharpest frame of a time window;
- routing a frame to restoration by its blur length.

**The corpus database is read-only.** A new score is stored beside it and never overwrites Update 1's.

## 4. Optionally: a new model on the distilled labels

A new model is trained only when the distilled signal is better than its teacher on the independent witnesses. A
new teacher has to earn that first.

**Update 2 is the pattern:**
- the same architecture, started again from ImageNet;
- soft cell targets;
- a measurement veto;
- a fixed readout.

**A next round could differ in three ways.** None of these is decided, and each step gets a registered protocol
first.
- **A less biased teacher:** Update 3, if it passes.
- **Length targets** in px, not only probabilities.
- **Consistency across codings.** The same frame coded twice must give the same map, and this needs no label. It
  applies only up to an acceptable quality: heavy compression is a real usability defect.

## The four parts in each update

| | Update 1 | Update 2 | Update 3 |
|---|---|---|---|
| dataset in training | 589,663 selected, VLM-labelled frames | every frame of the 167 training videos: 4,850,458 | as Update 1 |
| frame target | binary VLM label | Update 1's cell map (distillation) | binary VLM label |
| photo ladder | 0–3 / 16–30 px | 0–3 / 10–30 px | 0–30 px continuous, as a length |
| photo coding | single frame, H.264 (I-frame) | single frame, H.264 (I-frame) | 8-frame clip, x264 / x265 / AV1, I / IP / IPB |
| photo : frame | 1 : 1 | 1 : 3 | 1 : 1 |
| watermark masks | half of the photo and frame samples | half of the photo samples | none |
| distillation role | teacher: scored the 5.01 M frames | student of Update 1 | not yet: it is to score the corpus if it passes |
