"""Segment the teeth in panoramic dental X-rays.

For each input image, writes to --out:
  <name>.json         per tooth: number (universal 1-32), confidence, box, polygon (original pixels)
  <name>_labels.png   original size; each pixel = tooth number, 0 = background
  <name>_masks.npz    'masks': N x H x W boolean, one binary mask per tooth (same order as the JSON)
  <name>_overlay.jpg  the X-ray with the polygons drawn on, to check alignment by eye

Preprocessing matches evaluation exactly (src/common.py; tests/test_preprocess.py checks it):
8-bit 3-channel input, long side resized to the model's training size, gray padding, then
masks are cropped out of the padding and resized back to the original resolution.

Edge cases (all reported in the JSON 'warnings' or as a clear error, never a crash):
  missing / unreadable / too-small file  -> error for that image, others still processed
  colour, RGBA or 16-bit input           -> converted to 8-bit grayscale, with a warning
  no teeth found                         -> empty list and an all-zero mask
  overlapping masks                      -> each pixel goes to the more confident tooth
  tiny spurious masks                    -> dropped below --min-area-frac of the image area
  polygons                               -> largest outline per tooth, checked valid (>= 3 points,
                                            no self-intersections) with shapely

Examples:
    python predict.py xray.jpg --weights weights/best.pt
    python predict.py folder/ --weights weights/best.pt --conf-thresh 0.5 --out predictions/
    python predict.py folder/ --weights weights/best.pt --count-sweep 0.1,0.25,0.5,0.75
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.common import (IMGSZ, MIN_AREA_FRAC, PREDICT_CONF, VAL_IOU, is_color, letterbox,  # noqa: E402
                        load_image, tooth_number, unletterbox_mask)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LOCKED = Path(os.environ.get("POSTPROCESS_YAML", Path(__file__).resolve().parent / "configs" / "postprocess.yaml"))


def locked_thresholds():
    """conf/iou chosen on the validation set by src/sweep.py, if it has been run."""
    if LOCKED.exists():
        import yaml
        d = yaml.safe_load(LOCKED.read_text())
        return float(d.get("conf", PREDICT_CONF)), float(d.get("iou", VAL_IOU))
    return PREDICT_CONF, VAL_IOU


def mask_to_polygon(mask: np.ndarray):
    """Largest outer contour of a binary mask as a valid simple polygon [[x, y], ...], or None."""
    from shapely.geometry import Polygon
    from shapely.validation import make_valid

    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if len(c) >= 3]
    if not cnts:
        return None
    poly = Polygon(max(cnts, key=cv2.contourArea).reshape(-1, 2))
    if not poly.is_valid:
        def polygons(g):  # make_valid can nest: GeometryCollection > MultiPolygon > Polygon
            if g.geom_type == "Polygon":
                return [g]
            return [p for sub in getattr(g, "geoms", []) for p in polygons(sub)]
        parts = polygons(make_valid(poly))
        if not parts:
            return None
        poly = max(parts, key=lambda g: g.area)
    poly = poly.simplify(0.5, preserve_topology=True)
    if poly.is_empty or poly.geom_type != "Polygon" or not poly.is_valid or not poly.exterior.is_simple:
        return None
    coords = [[round(float(x), 1), round(float(y), 1)] for x, y in list(poly.exterior.coords)[:-1]]
    return coords if len(coords) >= 3 else None


def postprocess(masks, confs, classes, names, meta, conf_thresh=PREDICT_CONF, min_area_frac=MIN_AREA_FRAC):
    """Model output (masks in the letterboxed frame) -> teeth at the original resolution.

    masks: n x S x S (bool or 0-1 float), confs: n, classes: n. Pure function: no model needed.
    Returns (teeth list, n x H x W bool masks, label map H x W uint8, info dict).
    """
    h0, w0 = meta["orig_hw"]
    masks = np.asarray(masks).reshape(-1, meta["imgsz"], meta["imgsz"]) if len(masks) else np.zeros((0, 1, 1))
    confs, classes = np.asarray(confs, float).reshape(-1), np.asarray(classes, int).reshape(-1)
    keep = [i for i in range(len(confs)) if confs[i] >= conf_thresh]
    full = [unletterbox_mask(masks[i] > 0.5, meta) for i in keep]

    # overlaps: a pixel claimed by several teeth goes to the most confident one
    owner = np.full((h0, w0), -1, np.int32)
    best = np.full((h0, w0), -1.0, np.float32)
    claimed = np.zeros((h0, w0), np.uint8)
    for k, i in enumerate(keep):
        m = full[k]
        claimed += m.astype(np.uint8)
        win = m & (confs[i] > best)
        owner[win], best[win] = k, confs[i]
    info = {"overlap_pixels_resolved": int((claimed > 1).sum()), "dropped_small": 0, "dropped_invalid_polygon": 0}

    min_area = min_area_frac * h0 * w0
    teeth, out_masks = [], []
    for k, i in enumerate(keep):
        m = owner == k
        if m.sum() < max(min_area, 1):
            info["dropped_small"] += 1
            continue
        poly = mask_to_polygon(m)
        if poly is None:
            info["dropped_invalid_polygon"] += 1
            continue
        ys, xs = np.nonzero(m)
        name = names[int(classes[i])]
        teeth.append({"tooth": tooth_number(name), "label": str(name), "class_id": int(classes[i]),
                      "confidence": round(float(confs[i]), 4), "area_px": int(m.sum()),
                      "box_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                      "polygon": poly})
        out_masks.append(m)
    order = sorted(range(len(teeth)), key=lambda j: teeth[j]["tooth"])
    teeth = [teeth[j] for j in order]
    stack = np.stack([out_masks[j] for j in order]) if out_masks else np.zeros((0, h0, w0), bool)
    labels = np.zeros((h0, w0), np.uint8)
    for t, m in zip(teeth, stack):
        labels[m] = t["tooth"]
    return teeth, stack, labels, info


def predict_array(model, img, imgsz=IMGSZ, conf_thresh=PREDICT_CONF, iou=VAL_IOU, min_area_frac=MIN_AREA_FRAC):
    """Run the model on an image already in model format (H x W x 3 uint8)."""
    lb, meta = letterbox(img, imgsz)
    r = model.predict(lb, imgsz=imgsz, conf=conf_thresh, iou=iou, retina_masks=True, verbose=False)[0]
    if r.masks is None or len(r.boxes) == 0:
        masks, confs, classes = np.zeros((0, imgsz, imgsz)), [], []
    else:
        masks = r.masks.data.cpu().numpy()
        confs, classes = r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()
    return postprocess(masks, confs, classes, model.names, meta, conf_thresh, min_area_frac)


def predict_file(model, path, **kw):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    warnings = []
    if raw is not None:
        if raw.dtype == np.uint16:
            warnings.append("16-bit image converted to 8-bit")
        if raw.ndim == 3 and raw.shape[2] == 4:
            warnings.append("alpha channel dropped")
        if raw.ndim == 3 and raw.shape[2] >= 3 and is_color(raw[..., :3]):
            warnings.append("colour image converted to grayscale")
    img = load_image(path)
    teeth, masks, labels, info = predict_array(model, img, **kw)
    if not teeth:
        warnings.append("no teeth found above the confidence threshold")
    nums = [t["tooth"] for t in teeth]
    repeated = sorted({n for n in nums if nums.count(n) > 1})
    if repeated:
        warnings.append("tooth numbers predicted more than once (check for duplicates): "
                        + ", ".join(f"{n} x{nums.count(n)}" for n in repeated))
    h, w = img.shape[:2]
    result = {"image": str(path), "width": w, "height": h, "num_teeth": len(teeth),
              "conf_thresh": kw.get("conf_thresh", PREDICT_CONF), "iou": kw.get("iou", VAL_IOU),
              "imgsz": kw.get("imgsz", IMGSZ), "warnings": warnings, **info, "teeth": teeth}
    return result, img, masks, labels


def overlay(img, teeth):
    rng = np.random.default_rng(0)
    colors = {n: tuple(int(c) for c in rng.integers(60, 255, 3)) for n in range(40)}
    fill = img.copy()
    for t in teeth:
        cv2.fillPoly(fill, [np.round(t["polygon"]).astype(np.int32)], colors[t["tooth"]])
    out = cv2.addWeighted(fill, 0.4, img, 0.6, 0)
    k = max(img.shape[:2]) / 1500
    for t in teeth:
        cv2.polylines(out, [np.round(t["polygon"]).astype(np.int32)], True, colors[t["tooth"]], max(1, int(2 * k)))
        x0, y0 = t["box_xyxy"][:2]
        cv2.putText(out, f"{t['tooth']} {t['confidence']:.2f}", (x0, max(y0 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k, (255, 255, 255), max(1, int(2 * k)))
    return out


def save(result, img, masks, labels, out: Path):
    stem = Path(result["image"]).stem
    (out / f"{stem}.json").write_text(json.dumps(result, indent=1))
    cv2.imwrite(str(out / f"{stem}_labels.png"), labels)
    np.savez_compressed(out / f"{stem}_masks.npz", masks=masks)
    cv2.imwrite(str(out / f"{stem}_overlay.jpg"), overlay(img, result["teeth"]))


def list_images(source: Path):
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMG_EXTS)
    return [source]


def main(argv=None):
    conf0, iou0 = locked_thresholds()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="An image file or a folder of images")
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--conf-thresh", "--conf", dest="conf", type=float, default=conf0,
                    help=f"Minimum confidence to keep a tooth (default {conf0}, from configs/postprocess.yaml if present)")
    ap.add_argument("--iou", type=float, default=iou0, help=f"NMS IoU threshold (default {iou0})")
    ap.add_argument("--imgsz", type=int, default=None, help="Default: the size the weights were trained at")
    ap.add_argument("--min-area-frac", type=float, default=MIN_AREA_FRAC,
                    help="Drop masks smaller than this fraction of the image area")
    ap.add_argument("--count-sweep", default=None,
                    help="Comma-separated confidence thresholds: report teeth per image at each, save nothing else")
    args = ap.parse_args(argv)

    from ultralytics import YOLO
    from src.common import train_imgsz
    if not Path(args.weights).exists():
        sys.exit(f"Weights not found: {args.weights}")
    model = YOLO(args.weights)
    imgsz = args.imgsz or train_imgsz(args.weights)
    source = Path(args.source)
    if not source.exists():
        sys.exit(f"Input not found: {source}")
    images = list_images(source)
    if not images:
        sys.exit(f"No images ({', '.join(sorted(IMG_EXTS))}) in {source}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.count_sweep:
        ths = [float(x) for x in args.count_sweep.split(",")]
        rows = []
        for p in images:
            try:
                img = load_image(p)
            except (FileNotFoundError, ValueError) as e:
                print(f"skip {p.name}: {e}")
                continue
            rows.append([len(predict_array(model, img, imgsz, t, args.iou, args.min_area_frac)[0]) for t in ths])
        a = np.array(rows)
        lines = ["| Confidence threshold | Mean teeth per image | Min | Max |", "| --- | --- | --- | --- |"]
        lines += [f"| {t} | {a[:, j].mean():.1f} | {a[:, j].min()} | {a[:, j].max()} |" for j, t in enumerate(ths)]
        (out / "conf_counts.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return 0

    errors = {}
    for p in images:
        try:
            result, img, masks, labels = predict_file(model, p, imgsz=imgsz, conf_thresh=args.conf,
                                                      iou=args.iou, min_area_frac=args.min_area_frac)
            save(result, img, masks, labels, out)
            extra = f"  ({'; '.join(result['warnings'])})" if result["warnings"] else ""
            print(f"{p.name}: {result['num_teeth']} teeth -> {[t['tooth'] for t in result['teeth']]}{extra}")
        except (FileNotFoundError, ValueError) as e:
            errors[p.name] = str(e)
            print(f"ERROR {p.name}: {e}")
    if errors:
        (out / "errors.json").write_text(json.dumps(errors, indent=1))
    print(f"Saved to {out.resolve()}  ({len(images) - len(errors)} ok, {len(errors)} failed)")
    return 1 if len(errors) == len(images) else 0


if __name__ == "__main__":
    sys.exit(main())
