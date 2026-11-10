## Failure taxonomy: 20 worst test images (seed 0, conf 0.5, NMS IoU 0.7)

| Failure type | Teeth (worst 20) | Images affected (worst 20) | Teeth (all 88 test) |
| --- | --- | --- | --- |
| wrong number (neighbour) | 70 | 19 | 123 |
| missed | 39 | 15 | 54 |
| poor boundary | 9 | 5 | 12 |
| merged neighbours | 6 | 3 | 9 |
| other: false positive (bone / sinus / background) | 3 | 3 | 4 |
| wrong number (other) | 1 | 1 | 2 |
| edge clipping | 0 | 0 | 0 |
| metallic / restoration | 0 | 0 | 0 |
| wrong number (mirror) | 0 | 0 | 0 |

Metal and edge flags are heuristics: confirm in gallery_worst20.jpg and note corrections in failures_worst20.csv (manual_notes).

## Estimated mAP impact of the top two failure types

| Failure type | Images removed | mAP@50:95 all test | mAP@50:95 without them | Gain | Share of total mAP loss |
| --- | --- | --- | --- | --- | --- |
| wrong number (neighbour) | 26 of 88 | 0.594 | 0.612 | +0.017 ± 0.001 | 4% |
| missed | 6 of 88 | 0.594 | 0.604 | +0.010 ± 0.001 | 2% |

Gain = mAP without those images minus mAP on all test images (mean ± SD over seeds); share = gain / (1 - mAP). Rough: removing an image also removes its correctly segmented teeth.
