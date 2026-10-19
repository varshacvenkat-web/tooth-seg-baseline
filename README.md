# Tooth segmentation baseline

Instance segmentation of teeth on panoramic dental X-rays with YOLO11s-seg.
Each tooth gets its own mask and its universal tooth number (1–32).

**Data:** [Teeth Segmentation on Dental X-ray Images](https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images) (Kaggle, CC0), 598 images, split 70/15/15 → 418 train / 89 val / 91 test.

## Results (test split, mask metrics)

| Setup | Seeds | mAP@50 (mean ± sd) | mAP@50:95 (mean ± sd) |
| --- | --- | --- | --- |
| Baseline | 0, 1, 2 | _filled in by evaluate.py_ | |
| CLAHE | 0, 1, 2 | | |

Any difference smaller than the seed-to-seed spread is treated as noise.

## Settings

Ultralytics defaults except `epochs=50`. Fixed: `imgsz=640`, `batch=16`, `deterministic=True`,
COCO-pretrained `yolo11s-seg.pt`, optimizer `auto` (AdamW, lr ≈ 0.00027 for this run length).
Split seed 42; the exact file lists are in `splits.json`, written by `convert.py`.
Ultralytics saves every setting of each run in `runs/<name>/args.yaml`.

## Use the model

```bash
pip install -r requirements.txt
python predict.py path/to/xray.jpg --weights weights/best.pt --out predictions/
```

For each image `predict.py` writes:

| File | Contents |
| --- | --- |
| `<name>.json` | Per tooth: `tooth` (1–32), `confidence`, `box_xyxy`, `polygon` (pixel coordinates) |
| `<name>_mask.png` | Same size as the X-ray; pixel value = tooth number, 0 = background |
| `<name>_overlay.jpg` | The X-ray with predicted masks and numbers drawn on |

Weights are not stored in git; see the shared Drive folder.

## Reproduce

```bash
pip install -r requirements.txt
kaggle datasets download -d humansintheloop/teeth-segmentation-on-dental-x-ray-images -p data --unzip

# 1. Convert masks to YOLO polygons and split (seed 42)
python src/convert.py --src "data/Teeth Segmentation JSON" --out data/teeth_yolo

# 2. Train the baseline with three seeds
for s in 0 1 2; do python src/train.py --data data/teeth_yolo/data.yaml --seed $s --name baseline_s$s; done

# 3. Score on the test split: mean ± spread across seeds
python src/evaluate.py --data data/teeth_yolo/data.yaml --out results/baseline.csv \
  --runs baseline=runs/baseline_s0/weights/best.pt baseline=runs/baseline_s1/weights/best.pt baseline=runs/baseline_s2/weights/best.pt

# 4. Failure gallery: the 10 worst test images
python src/failures.py --weights runs/baseline_s0/weights/best.pt --data data/teeth_yolo --out results/failures

# 5. CLAHE experiment: same split (reused splits.json), same settings, same seeds
python src/convert.py --src <clahe copy in the Kaggle layout> --out data/teeth_yolo_clahe \
  --split-file data/teeth_yolo/splits.json
for s in 0 1 2; do python src/train.py --data data/teeth_yolo_clahe/data.yaml --seed $s --name clahe_s$s; done
python src/evaluate.py --data data/teeth_yolo_clahe/data.yaml --out results/clahe.csv \
  --runs clahe=runs/clahe_s0/weights/best.pt clahe=runs/clahe_s1/weights/best.pt clahe=runs/clahe_s2/weights/best.pt
```

## Repository layout

```
predict.py          inference on new images → JSON + mask PNG + overlay
src/convert.py      Kaggle masks → YOLO polygons, 70/15/15 split, splits.json
src/train.py        training with the fixed baseline settings
src/evaluate.py     test-split metrics per run, mean ± spread per group
src/failures.py     per-image scoring, failure categories, worst-10 gallery
week2_colab.ipynb   runs everything above in Google Colab
```
