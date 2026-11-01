# Update 1 and Update 3 on the whole corpus

Run on 2026-11-01, after Update 3 had scored the corpus. Every number comes from the scores and cell maps stored in the
private corpus database, read-only, with no new inference. The scripts in [eval/](eval/) cannot be rerun outside the
author's machine; they are published as the record of what was computed. The aggregate results are in [data/](data/),
the plan in [PLAN.md](PLAN.md) (the Hungarian original in [PLAN_hu.md](PLAN_hu.md)).

**In brief:** the two models make nearly the same frame decisions, and no witness picks a winner. The largest difference
is what the frame score is made of. Update 3's score also responds to the wide field of cells below logit 0, while
Update 1 Base's score comes almost only from the red cells (logit ≥ 0).

## What was compared

- **Frames:** all 5,010,602 corpus frames, from 173 videos:
  - 160,142 frames of the 6 held-out films;
  - 589,663 training frames (both models use the same frame index);
  - 4,260,797 other frames of the training films.
- **Models:** Update 1 Base (release v1.0 of the Update 1 repository) and Update 3 (ConvNeXt-Small). Update 1 Small has no
  corpus scores, so this is not the control used in the main [report](../u3r3_gop_ladder/REPORT.md).
- **Not a controlled comparison.** Base with recipe 1 against Small with recipe 3: the architecture and the recipe
  cannot be separated.
- **Neither model is a reference.** The first question is how they differ, then what the witnesses say.
- **Check on the data:** the validation AUC recomputed from the stored scores is 0.96207 and 0.96173; the runs recorded
  0.96206 and 0.96172.

**Terms used below:**

- **Equal sharp share:** per film, Update 3 keeps exactly as many frames as Update 1 keeps at that point. This removes
  the threshold difference.
- **Tiers,** proposed by the author, each model at its own thresholds: **A** below r98, **B** r98–r95, **C** r95–r90,
  **D** from r90.
- **Zero shot:** a shot from which the motion selection kept no frame.

## Agreement

| | all frames | held-out films | training frames | other frames |
|---|---|---|---|---|
| Spearman of the frame logits | 0.938 | 0.950 | 0.955 | 0.933 |
| same decision at r95 | 93.7% | 93.6% | 96.3% | 93.3% |
| κ at r95 | 0.869 | 0.870 | 0.926 | 0.860 |

- Inside a film the median Spearman is 0.932, inside a shot 0.861 (shots of at least 8 frames).
- At equal sharp share the r95 agreement is 93.9% (κ 0.872), so little of the disagreement is the threshold.
- **Strong contradictions** (one model A, the other D): 10,057 frames Update 1 A and Update 3 D, 6,825 the other way,
  3.4 per thousand in all. Motion does not explain them: their median motion rank inside the shot is 0.48 and 0.47.
- The rank difference inside a film follows no measured image statistic well. The strongest is blockiness, +0.09
  [0.08, 0.10]: Update 3 ranks blockier frames relatively blurrier, in 158 of 173 films.

## Thresholds

| | Update 1 Base | Update 3 | difference, median over films |
|---|---|---|---|
| sharp at r90 | 67.0% | 67.4% | +0.26 points [0.11, 0.49] |
| sharp at r95 | 59.9% | 58.8% | −0.95 [−1.19, −0.79] |
| sharp at r98 | 50.9% | 48.5% | −2.03 [−2.43, −1.72] |
| between r98 and r90 | 16.1% | 18.9% | +2.46 [2.29, 2.73], 168 of 173 films |
| shots with at least one A frame (48,392 shots) | 71.2% | 67.6% | – |

Update 3 is stricter at r98 and leaves more frames in the middle.

## Tiers

| tier | share of frames, Update 1 / Update 3 | `bsb` + `bbb` among labelled frames | `sss` among labelled frames | from zero shots |
|---|---|---|---|---|
| A | 50.9 / 48.5% | 0.8 / 0.6% | 56.1 / 56.7% | 10.6 / 9.7% |
| B | 9.0 / 10.3% | 5.1 / 4.2% | 40.4 / 42.0% | 22.6 / 22.5% |
| C | 7.2 / 8.6% | 13.0 / 12.0% | 32.8 / 34.0% | 29.8 / 29.4% |
| D | 33.0 / 32.6% | 71.4 / 71.8% | 6.7 / 6.5% | 48.2 / 48.6% |

The labelled frames (1,019,198) are not a random sample of the corpus; the codes are a gross check.

The tiers form a usable scale. The extremes are stable between the models, the middle is not:

| Update 1 tier | same tier in Update 3 |
|---|---|
| A | 89.6% |
| B | 43.8% |
| C | 46.5% |
| D | 91.4% |

83.0% of all frames are in the same tier, and 2.5% are two or more tiers apart.

## What carries the frame score

*Added after the results, 300,000 random frames.* The frame score with the red cells contributing nothing, against the
model's own r98:

| frames that stay above r98 without their red cells | Update 1 Base | Update 3 |
|---|---|---|
| tier B | 0.9% | 29.4% |
| tier C | 20.1% | 60.5% |
| tier C, zero shots | 32.9% | 80.5% |

- The median share of the score carried by red cells is 80% against 53% in tier B, and 88% against 77% in tier C.
- Cells at logit −6 or above cover a median 16.6% against 38.6% of a tier C frame.
- The median cell is −8.5 for Update 3. For Update 1 it is below the −16 at which the maps are stored (64% of its cells
  sit there).
- The frame rule is nearly the same in both, log-mean-exp with r = 0.86 and 0.70. Swapping the r values barely changes
  the agreement (Spearman 0.936–0.939 against 0.938). The difference is in the maps.
- A reading based only on red cells was not tested.

## Cell maps

- **Both models blurred at r95:** the maps agree. The within-frame Spearman is 0.665; in tier D of both, the red cells overlap with a
  median IoU of 0.66.
- **Both sharp:** the Spearman is 0.202. In tier A of both, where both have red cells, the median IoU is 0.06, and 38.7% of
  these frames have no common red cell.
- **Where the red cells sit** (*added after the results, the same sample*): the percent rank, inside the frame, of the
  measured motion of the 160 px blocks holding them. In tier A of both models it is 0.83 and 0.82: both put their few red
  cells on the most moving parts, though rarely on the same cells. In tier D it is 0.67 and 0.64.
- Frames with no red cell at all: 21.5% and 25.5%.

## Witnesses

At equal sharp share where a threshold is involved. The differences are medians of per-film differences (held-out column: per-shot
differences or a bootstrap over shots), so they need not equal the difference of the two columns. Intervals: 95%, 2,000
resamples. The held-out intervals cover these films' shots, not other films.

| witness | Update 1 Base | Update 3 | difference, 173 films | difference, 6 held-out films |
|---|---|---|---|---|
| frame logit against measured motion, inside a shot | 0.539 | 0.554 | +0.014 [0.012, 0.018] | +0.004 [−0.000, +0.008] |
| decision flips per 1,000 neighbouring pairs, r90 | 97.9 | 95.3 | −2.58 [−3.21, −1.94] | −3.92 [−6.66, −1.09] |
| the same, r95 | 101.8 | 100.6 | −0.96 [−2.12, −0.52] | −3.34 [−7.03, +0.06] |
| the same, r98 | 106.5 | 105.5 | −1.41 [−2.47, −0.10] | −1.73 [−5.33, +1.76] |
| motion of the kept frames at r95, median px | 4.96 | 4.94 | −0.02 [−0.03, −0.02] | – |
| motion rank of the frames only one model keeps, r95 (153,583 frames each) | 0.520 | 0.525 | −0.002 [−0.015, +0.010] | – |
| two releases of a film, all 907,218 matched pairs of 31 films: Spearman | 0.805 | 0.824 | +0.015 [0.010, 0.018] | – |
| two releases: r95 agreement | 85.01% | 84.95% | +0.02 points [−0.14, +0.18] | – |
| two releases: logit of the lower bitrate copy minus the higher | −0.056 | +0.001 | +0.052 [0.035, 0.108] | – |
| whole cell map against measured motion, inside a frame | 0.239 | 0.175 | −0.066 [−0.073, −0.060] | −0.075 [−0.084, −0.068] |
| where the red cells sit, tier A of both | 0.830 | 0.823 | −0.008 [−0.012, −0.005] | – |
| where the red cells sit, tier D of both | 0.674 | 0.637 | −0.035 [−0.037, −0.033] | – |
| VLM gross check: AUC on the labelled validation and test frames | 0.9685 | 0.9681 | – | −0.0003 [−0.0025, +0.0016] |

- **On the frame decision no witness picks a winner.** Update 3 is slightly ahead on motion and stability over the
  corpus. On the held-out films those intervals include 0, except at r90.
- **The frames the two models dispute show no difference** in motion. The VLM codes barely differ either: `bsb` + `bbb` is
  8.8% among the frames only Update 1 keeps, 7.8% among those only Update 3 keeps.
- **Both models flip about 10% of neighbouring frames** at a threshold.
- **The whole map follows motion better in Update 1 Base,** as it did in Update 1 Small (main report, item 4). In their
  red cells on sharp frames the two are nearly level.

## Inside tier C

*Added after the results.* Tier C of Update 3 holds 429,498 frames: 34.0% `sss` and 12.0% `bsb` + `bbb`.

- **The zero-shot flag does not separate it:** `bsb` + `bbb` 12.4% in zero shots and 11.7% in the others, `sss` 34.0% in both.
- **Being the best tier of its shot is not a rescue signal.** Where C is the best tier (7.2% of shots, 11,699 frames),
  `bsb` + `bbb` is higher: 21.6% in zero shots, 15.0% in the others, against 11.7% where the shot has a better tier.
- **Update 1's tier does separate it:**

  | Update 1 tier of Update 3's tier C frames | share | `bsb` + `bbb` | `sss` |
  |---|---|---|---|
  | A or B | 35% | 7.3% | 38.7% |
  | C | 39% | 12.3% | 34.4% |
  | D | 26% | 17.7% | 27.2% |

  This lowers the uncertainty of a frame at the price of a second model pass. It is not a measure of the view.

## High bitrate

*Added after the results.* Five films are above 12 Mbps; the median is 6.55. The three above 23 Mbps:

| film (Mbps, rounded) | sharp at r98 | sharp at r90 | median share of cells at logit −6 or above | logit against motion, inside the film |
|---|---|---|---|---|
| 39 | 60.6 / 31.8% | 74.7 / 74.0% | 6.4 / 58.5% | 0.485 / 0.254 |
| 25 | 53.2 / 49.0% | 70.2 / 71.6% | 7.0 / 29.3% | 0.587 / 0.523 |
| 24 | 56.7 / 45.4% | 70.0 / 70.1% | 5.9 / 40.9% | 0.576 / 0.540 |

Update 1 Base / Update 3. Update 3's field widens and its r98 share falls, while r90 does not move. Under 12 Mbps the
shift does not follow the bitrate (Spearman −0.065 over 168 films). Three films are a lead, not a result.

## The author's examples

- **A 104-frame extreme close-up shot.** Both models keep 7 frames at r90, with the tiers A / B / C / D at 1 / 2 / 4 / 97 for Update
  1 Base and 0 / 2 / 5 / 97 for Update 3. The author accepts Update 3's seven, and Update 1 places two of them in D.
  Without its red cells, none of the seven reaches A in Update 3. In Update 1 one of the seven is A, and three more would be.
- **A frame the author judges blurred, with a small blurred area.** Both models call it A. Update 3 marks 73 red cells, Update
  1 Base 20. The verdict is the same.

These are single cases. They show the mechanism above; they do not measure it.

## Limits

- Base with recipe 1 against Small with recipe 3: the causes cannot be separated.
- Frame types were not measured: the source videos were not available, and the frames carry no stored type. The codec
  results stay those of the main report.
- Motion is not blur. The VLM codes are a gross check, not a measure of validity.
- The zero-shot flag is a weak proxy for close-ups: 59% of the shots but 25% of the frames, with many short shots. No view
  label exists.
- Update 1's stored map is clipped at −16, with many ties at the bottom. This affects the whole-map witness, in an
  unknown direction.

## Planned and added

- **Planned before the results:** agreement, thresholds, confidence, contradictions, maps, witnesses a–f and the tiers ([PLAN.md](PLAN.md)). The plan was
  written to the local experiment log before the extraction; there is no outside timestamp for it. The log gives 23:35, typed by hand.
  The extraction started at about 23:33, so the entry was written earlier than its stamp says.
- **Late:** the plan's per-film values and shot bootstrap for the held-out films were left out of the first report script, and the log at first wrongly
  said there was no deviation. They were computed after the first results had been read, with the method unchanged
  ([eval/corpus_u1_u3_heldout.py](eval/corpus_u1_u3_heldout.py)).
- **Added after the results, not planned:** what carries the frame score, where the red cells sit, the tier pairs and tier C, the films with the largest
  differences, high bitrate, and the author's examples.

## Files

- [eval/corpus_u1_u3.py](eval/corpus_u1_u3.py): the per-frame extraction.
- [eval/corpus_u1_u3_report.py](eval/corpus_u1_u3_report.py): the planned tables, [data/corpus_u1_u3.json](data/corpus_u1_u3.json).
- [eval/corpus_u1_u3_heldout.py](eval/corpus_u1_u3_heldout.py): the held-out films, [data/corpus_u1_u3_heldout.json](data/corpus_u1_u3_heldout.json).
- [eval/corpus_u1_u3_posthoc.py](eval/corpus_u1_u3_posthoc.py): the additions, [data/corpus_u1_u3_posthoc.json](data/corpus_u1_u3_posthoc.json).
- [eval/corpus_u1_u3_bitrate.py](eval/corpus_u1_u3_bitrate.py): high bitrate, [data/corpus_u1_u3_bitrate.json](data/corpus_u1_u3_bitrate.json).
- [MANIFEST.json](MANIFEST.json): file hashes and the publication edits (film facts rounded or removed).
