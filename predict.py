"""Segment the teeth in dental panoramic X-rays and save the results.

For every input image this writes three files to --out:
  <name>.json         one entry per tooth: tooth number (universal 1-32),
                      confidence, box and outline polygon in pixel coordinates
  <name>_mask.png     same size as the X-ray; each pixel = tooth number, 0 = background
  <name>_overlay.jpg  the X-ray with the predicted masks and numbers drawn on

Examples:
    python predict.py xray.jpg --weights weights/best.pt
    python predict.py folder_of_xrays/ --weights weights/best.pt --out predictions/ --conf 0.25
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def tooth_number(name: str) -> int:
    """'14' -> 14, '13(polygon)' -> 13."""
    m = re.match(r"\d+", str(name))
    return int(m.group()) if m else 0


def predict_image(model, image_path, conf=0.25, imgsz=640):
    """Run the model on one image and return a JSON-ready dict."""
    r = model.predict(str(image_path), conf=conf, imgsz=imgsz, verbose=False)[0]
    h, w = r.orig_shape
    teeth = []
    if r.masks is not None:
        for poly, cls, cf, box in zip(r.masks.xy, r.boxes.cls.int().tolist(),
                                      r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
            if len(poly) < 3:
                continue
            name = model.names[cls]
            teeth.append({
                "tooth": tooth_number(name),
                "label": name,
                "class_id": cls,
                "confidence": round(cf, 4),
                "box_xyxy": [round(v, 1) for v in box],
                "polygon": [[round(float(x), 1), round(float(y), 1)] for x, y in poly],
            })
    teeth.sort(key=lambda t: t["tooth"])
    return {"image": str(image_path), "width": w, "height": h, "teeth": teeth}


def tooth_mask(result):
    """Label map: pixel = tooth number. Higher-confidence teeth win where masks overlap."""
    mask = np.zeros((result["height"], result["width"]), np.uint8)
    for t in sorted(result["teeth"], key=lambda t: t["confidence"]):
        cv2.fillPoly(mask, [np.round(t["polygon"]).astype(np.int32)], int(t["tooth"]))
    return mask


def overlay(image, result):
    rng = np.random.default_rng(0)
    colors = {n: tuple(int(c) for c in rng.integers(60, 255, 3)) for n in range(33)}
    fill = image.copy()
    for t in result["teeth"]:
        cv2.fillPoly(fill, [np.round(t["polygon"]).astype(np.int32)], colors[t["tooth"]])
    out = cv2.addWeighted(fill, 0.45, image, 0.55, 0)
    scale = max(image.shape[:2]) / 1500
    for t in result["teeth"]:
        x0, y0 = int(t["box_xyxy"][0]), int(t["box_xyxy"][1])
        cv2.putText(out, f"{t['tooth']} {t['confidence']:.2f}", (x0, max(y0 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (255, 255, 255), max(1, int(2 * scale)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="An image file or a folder of images")
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--conf", type=float, default=0.25, help="Minimum confidence to keep a tooth")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    src = Path(args.source)
    images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS) if src.is_dir() else [src]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    for p in images:
        result = predict_image(model, p, args.conf, args.imgsz)
        (out / f"{p.stem}.json").write_text(json.dumps(result, indent=1))
        cv2.imwrite(str(out / f"{p.stem}_mask.png"), tooth_mask(result))
        cv2.imwrite(str(out / f"{p.stem}_overlay.jpg"), overlay(cv2.imread(str(p)), result))
        nums = [t["tooth"] for t in result["teeth"]]
        print(f"{p.name}: {len(nums)} teeth -> {nums}")
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
