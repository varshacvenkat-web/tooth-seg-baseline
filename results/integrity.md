## Split integrity

Split sizes: train 418, val 89, test 88.

- File names shared by train and val: 0
- File names shared by train and test: 0
- File names shared by val and test: 0
- Byte-identical images across splits: 0
- Byte-identical images within one split (no leakage, noted): 0
- Image pairs across splits compared (contrast-equalised): 81818; image similarity median 0.55, 99.9th percentile 0.83
- Likely re-exports of the same radiograph across splits (image >= 0.90 and text strip >= 0.80): 0
- Duplicate groups in the whole dataset (any split): 6 groups, 12 images (saved to duplicate_groups.json)
- The 24 most similar cross-split pairs are in near_duplicates.jpg (all scores in near_duplicates.csv): same scan = same burned-in date/time, teeth, fillings and gaps.

**No leakage found** at the file and image level. The Kaggle export carries no patient identifiers, so images of the same patient taken on different visits cannot be grouped and could not be checked; this is a stated limitation. Cross-source generalisation is untested: all images come from one dataset; a second source (e.g. the Tufts Dental Database, by access request) would be needed.

## Baseline configuration audit

| Setting | baseline_s0 | baseline_s1 | baseline_s2 |
| --- | --- | --- | --- |
| model | yolo11s-seg.pt | yolo11s-seg.pt | yolo11s-seg.pt |
| imgsz | 640 | 640 | 640 |
| epochs | 50 | 50 | 50 |
| batch | 16 | 16 | 16 |
| optimizer | auto | auto | auto |
| lr0 | 0.01 | 0.01 | 0.01 |
| seed | 0 | 1 | 2 |
| deterministic | True | True | True |
| iou | 0.7 | 0.7 | 0.7 |
| conf | None | None | None |
| split | val | val | val |
| fliplr | 0.5 | 0.5 | 0.5 |
| mosaic | 1.0 | 1.0 | 1.0 |
| close_mosaic | 10 | 10 | 10 |
| hsv_v | 0.4 | 0.4 | 0.4 |
| scale | 0.5 | 0.5 | 0.5 |
| amp | True | True | True |
| data | /content/teeth_yolo/data.yaml | /content/teeth_yolo/data.yaml | /content/teeth_yolo/data.yaml |
| peak learning rate used (results.csv) | 0.000256 | 0.000256 | 0.000256 |

- Checkpoint selection split: val (validation only: the test split never chose a checkpoint).
- Input: long side resized to imgsz, aspect kept, padded with gray 114; pixels scaled to 0-1, BGR->RGB (Ultralytics; predict.py uses the same, see tests).
- Anchors: none. YOLO11 heads are anchor-free, so there are no anchor settings to tune.
- NMS IoU during validation = the iou row above (0.7 default); confidence 0.001 for mAP.
- optimizer 'auto' picks AdamW for short runs and ignores lr0; the peak rate actually used is shown.
