## Robustness: baseline on degraded test images (mean ± SD over 3 seeds)

| Test set | Mask mAP@50 | Mask mAP@50:95 | Drop in mAP@50:95 vs clean |
| --- | --- | --- | --- |
| clean | 0.950 ± 0.001 | 0.593 ± 0.003 | — |
| blur sigma 1.5 | 0.940 ± 0.002 | 0.569 ± 0.000 | 0.024 ± 0.003 (4%) |
| blur sigma 3 | 0.860 ± 0.029 | 0.477 ± 0.019 | 0.116 ± 0.022 (20%) |
| noise sigma 10 | 0.299 ± 0.066 | 0.132 ± 0.032 | 0.462 ± 0.029 (78%) |
| noise sigma 25 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.593 ± 0.003 (100%) |
| contrast 60% | 0.949 ± 0.004 | 0.590 ± 0.004 | 0.003 ± 0.003 (1%) |
| contrast 30% | 0.939 ± 0.004 | 0.569 ± 0.008 | 0.024 ± 0.006 (4%) |

Degradations applied after resizing to the 640 px model input; 'clean' = same resize, no degradation.
