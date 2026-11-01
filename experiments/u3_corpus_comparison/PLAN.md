# The plan, recorded before the results

English translation of [PLAN_hu.md](PLAN_hu.md), the two entries of the author's experiment log. The results were first
computed at 23:47 that evening.

## 2026-11-01 23:35: Update 1 and Update 3 on the whole corpus, analysis plan (before the result)

The stamp was typed by hand. The extraction started at about 23:33, so this entry was written earlier than it says.

- **The request:** the scoring of the corpus has finished; now the comparison of Update 1 and Update 3 on the 5.01 million frames.
- **Material:** 5,010,602 frames; both models have every score and cell map in the database (read-only). Update 1 = the
  v1.0 Base (`9840160f…`), Update 3 = the Small (`6cd68d83…`). Update 1 Small (v1.1) has no corpus scores.
- **Limits, in advance:**
  - The architecture and the recipe move together (Base + recipe 1 against Small + recipe 3), so the difference cannot be separated.
  - No frame types: the source videos are not available now, and the frames have no stored type. The codec dependence cannot be measured on the corpus.
  - Both models learned from the same index, so the groups are: the 6 held-out films, the trained frames, the rest.
  - Neither model is a reference, the motion measurement measures motion, not blur, and the VLM code is only a gross check.
- **The differences first, not the winner:**
  1. *Threshold:* sharp share at each model's own r90 / r95 / r98, by group and by film.
  2. *Confidence:* the distribution of the logits, the share of frames between a model's own r98 and r90.
  3. *Agreement:* Spearman globally, inside a film and inside a shot (at least 8 frames); the two-by-two table and κ at the
     own thresholds, and at equal sharp share per film (Update 3's threshold is set per film to keep as many frames as Update 1).
  4. *Strong contradiction:* one under its own r98, the other above its own r90.
  5. *Map:* cell by cell, on every frame: the within-frame Spearman of the two maps, red share (logit ≥ 0), the overlap of the
     red cells (IoU), the share with M = 0, the largest red blob. The own `r` values (Update 1 about 0.86, Update 3 about 0.70)
     swapped: does the difference come from the map or from the readout?
- **Witnesses, at equal sharp share:**
  - a. the measured motion of the kept frames, the within-shot Spearman of logit and motion;
  - b. the motion rank inside the shot of the frames only one model keeps;
  - c. stability over time: decision flips between neighbouring frames of a shot, lag-1 autocorrelation;
  - d. on every frame that can be paired in the films released twice: the Spearman of the logits, the r95 agreement and κ;
  - e. the cell-level motion witness (R4T-1 9.6) from the stored maps, on every frame with a measurement;
  - f. gross check: the VLM codes among the kept frames, AUC on the held-out films.
- **Summary:** for the differences (Update 3 − Update 1) a bootstrap over films (2000, seed 20261030); for the 6 held-out films the per-film values and a bootstrap over shots.
- **Reading, in advance:** direction and interval reported per witness; no overall winner is declared. If a witness shows a
  difference only in motion or only on the trained frames, it is described that way.

## 2026-11-01 23:37: added to the comparison: the decision tiers (before the result)

- **The author's proposal:** the bands between the thresholds become categories: below r98, r98–r95, r95–r90 ("sharp, less
  sharp, least sharp"). The extreme close-ups that come in, and the blurred close-ups, are to be looked for in the least sharp
  band. The names `sss`, `ss`, `s` would collide with the VLM codes, so here **A** (below r98), **B** (r98–r95), **C** (r95–r90), **D** (from r90).
- **Measured for both models:** the share of the tiers by group and by film; the table of the tiers against each other
  (Update 1 × Update 3); per tier the share of zero shots, the VLM codes, the measured motion, the red share and the peak;
  the best tier of the shots; the frames of tier C split in two: the shot has a better tier, or C is the shot's best.
- **Limit:** there is no view label. The zero shot is only a gross flag for close-ups, and does not separate extreme close-ups from close-ups.
