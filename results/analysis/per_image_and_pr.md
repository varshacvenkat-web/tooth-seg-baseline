## Per-image mean mask IoU (test, 88 images, mean over seeds)

10th / 25th / 50th / 75th / 90th percentile: 0.569 / 0.687 / 0.759 / 0.813 / 0.838
Min 0.077, max 1.000, 6 images below 0.5.

Hardest: 355.jpg (0.08), 412.jpg (0.28), 9.jpg (0.33), 193.jpg (0.47), 128.jpg (0.47)
Easiest: 184.jpg (0.85), 301.jpg (0.85), 172.jpg (0.86), 227.jpg (0.86), 300.jpg (1.00)

## Precision vs recall at the locked operating point (conf 0.5, NMS IoU 0.7; match = same tooth number and IoU >= 0.5)

| Seed | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2044 | 202 | 200 | 0.910 | 0.911 | 0.910 |
| 1 | 2037 | 172 | 207 | 0.922 | 0.908 | 0.915 |
| 2 | 2001 | 163 | 243 | 0.925 | 0.892 | 0.908 |
| mean ± SD | | 179.000 ± 20.421 | 216.667 ± 23.072 | 0.919 ± 0.008 | 0.903 ± 0.010 | 0.911 ± 0.004 |

Errors are dominated by **false negatives (missed or mis-segmented teeth)**.

False negatives per seed, by cause: wrong number (neighbour) 116.3, missed 81.3, poor boundary 10.0, merged neighbours 8.0, wrong number (other) 1.0
False positives per seed, by kind: wrong_number 130.3, duplicate 25.7, poor_boundary 19.0, phantom 4.0
