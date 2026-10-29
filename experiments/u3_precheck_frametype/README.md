# Pre-check: do the VLM labels depend on the coded frame type?

Run on 2026-10-29, before the Update 3 run started. The code is [label_frametype.py](label_frametype.py), the result
is [data/label_frametype.json](data/label_frametype.json). The script reads the private corpus database (read-only)
and the source videos, so it cannot be rerun outside the author's machine. It is published as the record of what was
computed.

## Question

R4K-1 and R4K-2 (in the [Update 2 repository](https://github.com/mschauta/frame-blur-detection-update-2)) showed that
Update 1 scores an I-frame sharper than its P/B neighbours. If the VLM labels carried the same bias, the frame labels
would teach it to every model trained on them, whatever the photos do.

## Method

- **Frame types.** For each video of the corpus database, ffprobe lists the packets of the source file. The
  database lists 177 videos; the 173 videos of the corpus hold its frames. A packet with the key flag (K) marks a
  keyframe; the corpus frames carry the source timestamp.
- **Labels.** 1,019,198 of the 1,019,201 labelled frames carry one of the six codes used (`sss`, `ssb`, `bss`,
  `sbb`, `bsb`, `bbb`); the other three hold an unanswered question. 12,116 of them are keyframes. I-frames without
  the key flag count as non-I, so the null is slightly diluted.
- **Triples.** A keyframe is compared with its two neighbours (frame index − 1 and + 1, same scene, all three
  labelled). The null is the same comparison centred on non-I frames, a fixed-seed 5% sample.
- **Sharpness of a label:** the number of "sharp" answers in the three-question VLM code (sss = 3, bbb = 0).
- **The same comparison** on Update 1 Base's corpus score (blur probability) of the same frames.

## Result

| | keyframe in the centre | non-I in the centre (null) |
|---|---|---|
| triples | 11,743 | 49,075 |
| VLM: centre sharper than both neighbours | 7.1% | 6.0% |
| VLM: centre blurrier than both neighbours | 5.2% | 5.3% |
| VLM: mean sharpness, centre − neighbours (0–3 scale) | +0.012 | +0.001 |
| Update 1 Base: centre has the lowest blur probability | 49.1% | 26.5% |
| Update 1 Base: median logit, centre − neighbours | −0.65 | +0.03 |

The VLM labels practically do not depend on the frame type. Update 1 Base does, over all 173 videos. The bias
therefore comes from the model, most likely from its uncoded photo anchors, and not from the frame labels.
