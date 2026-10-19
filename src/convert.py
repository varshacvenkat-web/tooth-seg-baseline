"""Convert the Kaggle "Teeth Segmentation on Dental X-ray Images" dataset
(Supervisely export) into YOLO segmentation format and split it.

Each mask PNG stores one tooth ID per pixel. For every tooth we isolate its
pixels, trace the outline with cv2.findContours, normalize the points to 0-1
and write one line per tooth: `class x1 y1 x2 y2 ...`.

The split is 70/15/15 with a fixed seed, and the exact file lists are saved to
splits.json. Pass --split-file to reuse a saved split (e.g. for the CLAHE copy),
so two datasets are guaranteed to use the same train/val/test images.

Example:
    python src/convert.py --src "data/Teeth Segmentation JSON" --out data/teeth_yolo
    python src/convert.py --src data/clahe_kaggle --out data/teeth_yolo_clahe \
        --split-file data/teeth_yolo/splits.json
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def find_mask(mask_dir: Path, img: Path):
    for cand in (mask_dir / f"{img.stem}.png", mask_dir / f"{img.name}.png"):
        if cand.exists():
            return cand
    return None


def apply_clahe(img, clip=2.0, tile=8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(gray)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def mask_to_lines(mask, val2cls, w, h, min_area=20):
    lines = []
    for v in np.unique(mask):
        if v not in val2cls:
            continue  # 0 = background
        binary = (mask == v).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < min_area or len(c) < 3:
                continue  # annotation specks
            pts = (c.reshape(-1, 2) / [w, h]).flatten()
            lines.append(f"{val2cls[v]} " + " ".join(f"{x:.6f}" for x in pts))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder with obj_class_to_machine_color.json and d2/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-file", help="Reuse an existing splits.json instead of shuffling")
    ap.add_argument("--clahe", action="store_true", help="Apply CLAHE to images while copying")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    img_dir, mask_dir = src / "d2" / "img", src / "d2" / "masks_machine"
    cmap = json.loads((src / "obj_class_to_machine_color.json").read_text())
    names = list(cmap.keys())
    val2cls = {cmap[n][0]: i for i, n in enumerate(names)}

    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    by_name = {p.name: p for p in imgs}

    if args.split_file:
        saved = json.loads(Path(args.split_file).read_text())
        splits = {k: [by_name[n] for n in v] for k, v in saved.items()}
    else:
        random.seed(args.seed)
        random.shuffle(imgs)
        n = len(imgs)
        n_tr, n_va = int(0.7 * n), int(0.15 * n)
        splits = {"train": imgs[:n_tr], "val": imgs[n_tr:n_tr + n_va], "test": imgs[n_tr + n_va:]}

    missing = resized = 0
    for split, files in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for p in files:
            mp = find_mask(mask_dir, p)
            if mp is None:
                missing += 1
                continue
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            mask = cv2.imread(str(mp))[..., 0]
            if mask.shape != (h, w):
                # nearest neighbor: copies the closest pixel, so tooth IDs never blend
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                resized += 1
            if args.clahe:
                cv2.imwrite(str(out / "images" / split / p.name), apply_clahe(img))
            else:
                shutil.copy(p, out / "images" / split / p.name)
            lines = mask_to_lines(mask, val2cls, w, h)
            (out / "labels" / split / f"{p.stem}.txt").write_text("\n".join(lines))

    (out / "splits.json").write_text(json.dumps({k: [p.name for p in v] for k, v in splits.items()}, indent=1))
    yaml = f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
    yaml += "".join(f"  {i}: '{nm}'\n" for i, nm in enumerate(names))
    (out / "data.yaml").write_text(yaml)
    print({k: len(v) for k, v in splits.items()}, "| missing masks:", missing, "| resized:", resized)


if __name__ == "__main__":
    main()
