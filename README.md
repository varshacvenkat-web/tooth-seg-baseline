# Tooth instance segmentation on panoramic dental X-rays

YOLO11s-seg finds every tooth in a panoramic X-ray, draws its mask and labels it with its
universal tooth number (1–32). This repo covers the baseline, its seed variance, a post-processing
sweep, per-class / per-image / failure analysis, one pre-registered experiment, robustness checks
and a validated `predict.py`.

**Data:** [Teeth Segmentation on Dental X-ray Images](https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images)
(Kaggle, CC0): 598 images, split 70/15/15 → 418 train / 89 val / 91 test (seed 42; exact lists in `splits.json`).

## Results

Week 4 baseline: YOLO11s-seg, 640 px, 50 epochs, seeds 0/1/2, **88-image test set** (3 test images
removed because they were re-exports of train/val scans; see Data integrity below). All numbers are
mean ± SD over the 3 seeds. Raw outputs are in [`results/`](results/).

| Setup | Seeds | Mask mAP@50 (mean ± SD) | Mask mAP@50:95 (mean ± SD) |
| --- | --- | --- | --- |
| Baseline | 0, 1, 2 | 0.9503 ± 0.0018 | 0.5942 ± 0.0038 |
| Tuned post-processing | 0, 1, 2 | 0.9503 ± 0.0018 | 0.5942 ± 0.0038 |
| Experiment 1 (`mosaic: 0.0`, pre-registered, pending sign-off) | 0, 1, 2 | | |

*Tuned post-processing* = the same baseline weights evaluated with the NMS IoU locked on the validation
set (`configs/postprocess.yaml`); mAP is computed over all confidences (0.001), and precision/recall at the
locked confidence are in `results/analysis/per_image_and_pr.md`. All rows: same test split, same seeds,
paired by seed; per-seed deltas, paired t-test, Wilcoxon and Cohen's d_z will be in `results/comparison.md`.

**Threshold sweep (val).** 4 confidences × 4 NMS IoUs; locked NMS IoU 0.7 (the default) and confidence 0.5
(val F1 0.887). Tuning changed nothing on test: an honest null result.
[`results/sweep/sweep_val.md`](results/sweep/sweep_val.md), examples in `results/sweep/nms_example_*.jpg`.

**Operating point (conf 0.5, test).** Precision 0.919 ± 0.008, recall 0.903 ± 0.010, F1 0.911 ± 0.004.
Errors are mostly false negatives, and the largest cause is off-by-one tooth numbering.

**By tooth type** ([`ap_by_type.md`](results/analysis/ap_by_type.md)). AP@50 is high for every type
(0.94 to 0.96). Precise outlines are the hard part: AP@85 is 0.090 for incisors vs 0.339 for 1st/2nd
molars. All teeth: AP@75 0.706, AP@85 0.185.

![AP vs IoU threshold by tooth type](results/analysis/ap_vs_iou_by_type.png)

**Per image** ([`per_image_and_pr.md`](results/analysis/per_image_and_pr.md)). Median mean-mask IoU 0.759;
6 of 88 images below 0.5. Hardest: 355, 412, 9, 193, 128.

![Per-image IoU histogram](results/analysis/per_image_iou_hist.png)

**Failure taxonomy** ([`failures.md`](results/failures/failures.md)), seed 0, 20 worst test images:

| Failure type | Teeth (worst 20) | Teeth (all 88) |
| --- | --- | --- |
| Wrong number, neighbour (off by one) | 70 | 123 |
| Missed | 39 | 54 |
| Poor boundary | 9 | 12 |
| Merged neighbours | 6 | 9 |
| False positive (bone / sinus / background) | 3 | 4 |
| Metallic / restoration, edge clipping | 0 | 0 |

Estimated mAP@50:95 impact: removing the images dominated by neighbour numbering errors gives +0.017 ± 0.001;
missed teeth, +0.010 ± 0.001. Visual catalog: `results/failures/gallery_worst20.jpg`, `catalog_*.jpg`,
`worst_*.jpg`.

![Worst 20 test images](results/failures/gallery_worst20.jpg)

**Robustness** ([`robustness.md`](results/robustness/robustness.md)), drop in mAP@50:95 vs clean:
blur σ1.5 4%, blur σ3 20%, contrast 60% 1%, contrast 30% 4%, Gaussian noise σ10 78%, σ25 100%.
The model tolerates blur and contrast changes but fails on noisy images.

**Data integrity** ([`integrity.md`](results/integrity.md)). No shared file names or byte-identical
images across splits. An all-pairs comparison after contrast equalisation found 3 radiographs exported
twice (same burned-in date/time) with the copies in different splits: train/250 ~ test/249,
train/26 ~ test/25, val/85 ~ test/86 (similarity 0.999 vs a 99.9th percentile of 0.83 across all
pairs). The 3 test copies are excluded (`configs/duplicate_groups.json`); train and val are unchanged,
so the trained models remain valid. Removing them lowered mAP@50 from 0.9524 to 0.9503. Patient-level
grouping across visits is not possible (no patient IDs in the export).

**predict.py demo** ([`demo.md`](results/demo/demo.md)). 10 unseen test images, all processed; masks
at the original resolution; edge cases (blank, colour, RGBA, 16-bit, corrupted, 20 px, missing file)
handled with a warning or a clear error. 33 unit tests pass (`results/pytest.txt`).

![predict.py on 10 unseen test images](results/demo/demo_contact_sheet.jpg)

## Setup

```bash
git clone https://github.com/varshacvenkat-web/tooth-seg-baseline && cd tooth-seg-baseline
pip install -r requirements.txt          # plus a CUDA build of torch for training
export KAGGLE_API_TOKEN=...              # only needed to download the dataset
```

In Colab, open `week4_colab.ipynb` instead: it does all of this and keeps runs on Google Drive.

## Reproduce everything (one command)

```bash
bash reproduce.sh
```

| Stage | Command | What it does | Main outputs |
| --- | --- | --- | --- |
| data | `bash reproduce.sh data` | Download, convert masks to YOLO polygons, split, leakage check | `data/teeth_yolo/`, `results/integrity.md` |
| baseline | `bash reproduce.sh baseline` | Train 3 seeds; test metrics; val threshold sweep; tuned test metrics | `results/baseline_test.md`, `results/sweep/`, `configs/postprocess.yaml` |
| analyze | `bash reproduce.sh analyze` | AP by tooth type and IoU; per-image IoU; precision/recall; 20-worst failure taxonomy and impact; degradation test; config audit | `results/analysis/`, `results/failures/`, `results/robustness/`, `results/integrity.md` |
| demo | `bash reproduce.sh demo` | `predict.py` on 10 unseen images + edge cases | `results/demo/` |
| tests | `bash reproduce.sh tests` | pytest | `results/pytest.txt` |
| exp1 | `bash reproduce.sh exp1` | Pre-registered experiment (needs `experiments/exp1.yaml` signed off), paired comparison | `results/comparison.md` |

Finished training runs are skipped on rerun. Options: `SEEDS="0 1 2 3 4"`, `RUNS_DIR=/path/to/runs`,
`DEMO_IMAGES=/path/to/new/xrays`. `SMOKE=1 bash reproduce.sh` runs the whole pipeline on a small fake
dataset on a CPU in a few minutes, to check the code before spending GPU time.

**Expected runtime** on a Colab T4: ~18 min per baseline seed at 640 px; the sweep, analysis and
degradation stages ~20–30 min together; an `imgsz=1024` experiment ~45 min per seed.

## Use the model

```bash
python predict.py path/to/xray.jpg --weights weights/best.pt --out predictions/
python predict.py folder/ --weights weights/best.pt --conf-thresh 0.5
python predict.py folder/ --weights weights/best.pt --count-sweep 0.1,0.25,0.5,0.75   # teeth per image vs threshold
```

| Output | Contents |
| --- | --- |
| `<name>.json` | Per tooth: `tooth` (1–32), `confidence`, `box_xyxy`, `polygon` (original pixel coordinates), `area_px`; plus warnings |
| `<name>_labels.png` | Original size; pixel value = tooth number, 0 = background |
| `<name>_masks.npz` | `masks`: N × H × W boolean, one binary mask per tooth, original resolution |
| `<name>_overlay.jpg` | The X-ray with the polygons drawn on (alignment check) |

Defaults for `--conf-thresh` and `--iou` come from `configs/postprocess.yaml` (locked on the
validation set by `src/sweep.py`). Weights are not stored in git: [download link].

**Preprocessing parity.** `predict.py` reads images exactly as Ultralytics does during evaluation
(8-bit, 3-channel BGR), resizes the long side to the training size with bilinear interpolation and pads
with gray 114; masks are cropped out of the padding and resized back to the original resolution.
`tests/test_preprocess.py::test_parity_with_ultralytics_eval_loader` fails if this ever drifts.

**Edge cases handled:** missing, corrupted or too-small files (clear error, other images still run);
colour, RGBA and 16-bit images (converted, with a warning); no teeth found (empty JSON, blank mask);
overlapping masks (each pixel to the more confident tooth); tiny spurious masks (`--min-area-frac`,
default 0.05% of the image); invalid polygons (repaired with shapely or dropped; never < 3 points).

## Repository layout

```
reproduce.sh              one command: data → baseline → analysis → demo → tests → exp1
predict.py                inference → JSON + label PNG + binary masks + overlay
src/common.py             settings and preprocessing shared by training, evaluation and predict.py
src/convert.py            Kaggle masks → YOLO polygons, split, splits.json
src/train.py              training with the locked settings; --set key=value for one experimental change
src/evaluate.py           per-seed test metrics, mean ± SD
src/sweep.py              confidence × NMS IoU sweep on val; locks configs/postprocess.yaml
src/imgeval.py            per-image, per-tooth matching and failure categories
src/analyze.py            AP by tooth type and IoU threshold, per-image IoU, precision/recall
src/failures.py           20-worst taxonomy, visual catalog, mAP impact estimate
src/compare.py            paired t-test, Wilcoxon, effect size, per-seed deltas
src/degrade.py            blur / noise / contrast stress test
src/check_integrity.py    split leakage check and baseline config audit
src/demo.py               predict.py on 10 unseen images + edge cases
experiments/              exp1.yaml (pre-registration) and how to fill it in
tests/                    pytest: preprocessing parity, un-padding, predict.py edge cases
```

## Known limitations

- No patient IDs in the dataset: splits are checked for duplicate and near-duplicate images, but two
  visits of the same patient could still land in different splits.
- One source only: generalisation to other scanners or clinics is untested.
- 3 seeds give a rough noise estimate; the Wilcoxon test cannot reach p < 0.05 with fewer than 6 seeds.
- Metal and field-of-view failure flags are image heuristics, checked by eye in the gallery.
- The dataset metadata lists a 33rd class, `13(polygon)`, with no instances; it does not affect mAP.
