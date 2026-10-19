"""Find the worst test images and save a failure gallery.

For every test image, each ground-truth tooth is compared with the
predictions (mask IoU, computed at reduced resolution for speed):
  ok            a prediction with the same tooth number overlaps it with IoU >= 0.5
  missed        no prediction of any number overlaps it (IoU < 0.3)
  wrong number  found (IoU >= 0.5) but labeled as another tooth; split into
                mirror (same tooth, other side) and neighbor (adjacent tooth)
  poor outline  everything else: the right area but a loose or partial mask
Also flagged per image: merged (one prediction covering 30%+ of two or more
teeth), teeth touching the image edge, and extra predictions matching nothing.

Image score = mean IoU of each ground-truth tooth with its best same-number
prediction (a missed tooth counts as 0). The 10 lowest scores are saved as
overlays: green = ground-truth outline, colored fill = prediction, red box =
a failed tooth with a tag (M missed, W wrong number, O poor outline, E edge).
Crowns, implants and fillings can't be detected automatically: look at the
gallery and add them in the manual_notes column of failures.csv.

Example:
    python src/failures.py --weights runs/baseline_s0/weights/best.pt \
        --data data/teeth_yolo --out results/failures
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict import predict_image, tooth_number  # noqa: E402

WORK_SIDE = 1024  # rasterize masks at this longest side for speed


def mirror(n):
    return 17 - n if n <= 16 else 49 - n


def neighbor(a, b):
    return abs(a - b) == 1 and (a <= 16) == (b <= 16)


def shape(polys, s, h, w):
    """Rasterize polygons (pixel coords) at scale s; keep the crop box for fast IoU."""
    m = np.zeros((int(h * s) + 2, int(w * s) + 2), np.uint8)
    allp = []
    for p in polys:
        q = np.round(np.asarray(p, float) * s).astype(np.int32)
        cv2.fillPoly(m, [q], 1)
        allp.append(q)
    q = np.concatenate(allp)
    x0, y0 = q.min(0)
    x1, y1 = q.max(0) + 1
    box = (max(x0, 0), max(y0, 0), x1, y1)
    return {"m": m, "box": box, "area": int(m.sum())}


def inter(a, b):
    x0, y0 = max(a["box"][0], b["box"][0]), max(a["box"][1], b["box"][1])
    x1, y1 = min(a["box"][2], b["box"][2]), min(a["box"][3], b["box"][3])
    if x1 <= x0 or y1 <= y0:
        return 0
    return int((a["m"][y0:y1, x0:x1] & b["m"][y0:y1, x0:x1]).sum())


def iou(a, b):
    i = inter(a, b)
    u = a["area"] + b["area"] - i
    return i / u if u else 0.0


def load_gt(label_file, names, w, h):
    gt = {}
    for line in label_file.read_text().splitlines():
        v = line.split()
        if len(v) < 7:
            continue
        pts = np.array(v[1:], float).reshape(-1, 2) * [w, h]
        gt.setdefault(tooth_number(names[int(v[0])]), []).append(pts)
    return gt


def analyze(img_path, label_file, model, conf):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    s = min(1.0, WORK_SIDE / max(h, w))
    gt_polys = load_gt(label_file, model.names, w, h)
    pred = predict_image(model, img_path, conf=conf)
    G = {n: shape(p, s, h, w) for n, p in gt_polys.items()}
    P = [(t, shape([t["polygon"]], s, h, w)) for t in pred["teeth"]]

    teeth, scores = [], []
    for n, g in G.items():
        same = max((iou(g, ps) for t, ps in P if t["tooth"] == n), default=0.0)
        best_any, best_n = 0.0, None
        for t, ps in P:
            v = iou(g, ps)
            if v > best_any:
                best_any, best_n = v, t["tooth"]
        bx = g["box"]
        edge = bool(bx[0] <= 0.01 * w * s or bx[2] >= 0.99 * w * s or bx[1] <= 0.01 * h * s or bx[3] >= 0.99 * h * s)
        if same >= 0.5:
            cat = "ok"
        elif best_any < 0.3:
            cat = "missed"
        elif best_any >= 0.5 and best_n != n:
            cat = "mirror" if best_n == mirror(n) else "neighbor" if neighbor(best_n, n) else "wrong number"
        else:
            cat = "poor outline"
        teeth.append({"tooth": n, "iou": same, "cat": cat, "edge": edge, "pred_as": best_n})
        scores.append(same)

    merged = sum(1 for _, ps in P if sum(inter(g, ps) >= 0.3 * g["area"] for g in G.values()) >= 2)
    extra = sum(1 for _, ps in P if max((iou(g, ps) for g in G.values()), default=0.0) < 0.3)
    cats = [t["cat"] for t in teeth]
    row = {
        "image": img_path.name,
        "score": round(float(np.mean(scores)) if scores else 0.0, 3),
        "gt_teeth": len(G),
        "pred_teeth": len(P),
        "missed": cats.count("missed"),
        "wrong_number": sum(c in ("mirror", "neighbor", "wrong number") for c in cats),
        "mirror": cats.count("mirror"),
        "neighbor": cats.count("neighbor"),
        "poor_outline": cats.count("poor outline"),
        "merged": merged,
        "edge_teeth_failed": sum(t["edge"] and t["cat"] != "ok" for t in teeth),
        "extra_predictions": extra,
        "manual_notes": "",
    }
    return row, img, gt_polys, pred, teeth


def draw(img, gt_polys, pred, teeth, row):
    out = img.copy()
    rng = np.random.default_rng(0)
    colors = {n: tuple(int(c) for c in rng.integers(60, 255, 3)) for n in range(33)}
    fill = out.copy()
    for t in pred["teeth"]:
        cv2.fillPoly(fill, [np.round(t["polygon"]).astype(np.int32)], colors[t["tooth"]])
    out = cv2.addWeighted(fill, 0.4, out, 0.6, 0)
    k = max(img.shape[:2]) / 1500
    for n, polys in gt_polys.items():
        for p in polys:
            cv2.polylines(out, [np.round(p).astype(np.int32)], True, (0, 220, 0), max(1, int(2 * k)))
    tags = {"missed": "M", "mirror": "W", "neighbor": "W", "wrong number": "W", "poor outline": "O"}
    for t in teeth:
        if t["cat"] == "ok":
            continue
        q = np.concatenate([np.round(p).astype(np.int32) for p in gt_polys[t["tooth"]]])
        x0, y0 = q.min(0)
        x1, y1 = q.max(0)
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), max(1, int(2 * k)))
        tag = f"{t['tooth']}{tags[t['cat']]}" + ("E" if t["edge"] else "")
        if t["cat"] in ("mirror", "neighbor", "wrong number"):
            tag += f"->{t['pred_as']}"
        cv2.putText(out, tag, (int(x0), int(y0) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, (0, 0, 255), max(1, int(2 * k)))
    head = (f"{row['image']}  score {row['score']:.2f}  missed {row['missed']}  wrong# {row['wrong_number']}  "
            f"outline {row['poor_outline']}  merged {row['merged']}  extra {row['extra_predictions']}")
    cv2.rectangle(out, (0, 0), (out.shape[1], int(44 * k)), (0, 0, 0), -1)
    cv2.putText(out, head, (int(10 * k), int(30 * k)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * k, (255, 255, 255), max(1, int(2 * k)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="YOLO dataset folder with images/test and labels/test")
    ap.add_argument("--out", default="results/failures")
    ap.add_argument("--worst", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    imgs = sorted((data / "images" / "test").iterdir())

    results = []
    for p in imgs:
        row, img, gt_polys, pred, teeth = analyze(p, data / "labels" / "test" / f"{p.stem}.txt", model, args.conf)
        results.append((row, img, gt_polys, pred, teeth))
        print(f"{row['image']}: score {row['score']:.3f}")

    results.sort(key=lambda r: (r[0]["score"], -r[0]["extra_predictions"]))
    with (out / "failures.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(results[0][0]))
        wr.writeheader()
        wr.writerows(r[0] for r in results)

    thumbs = []
    for rank, (row, img, gt_polys, pred, teeth) in enumerate(results[: args.worst], 1):
        vis = draw(img, gt_polys, pred, teeth, row)
        cv2.imwrite(str(out / f"worst_{rank:02d}_{Path(row['image']).stem}.jpg"), vis)
        thumbs.append(cv2.resize(vis, (800, int(800 * vis.shape[0] / vis.shape[1]))))
    th = max(t.shape[0] for t in thumbs)
    thumbs = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in thumbs]
    while len(thumbs) % 2:
        thumbs.append(np.zeros_like(thumbs[0]))
    rows = [np.hstack(thumbs[i:i + 2]) for i in range(0, len(thumbs), 2)]
    cv2.imwrite(str(out / "gallery.jpg"), np.vstack(rows))

    totals = {k: sum(r[0][k] for r in results[: args.worst]) for k in
              ("missed", "mirror", "neighbor", "wrong_number", "poor_outline", "merged", "edge_teeth_failed", "extra_predictions")}
    print(f"\nWorst {args.worst} images, failure counts: {totals}")
    print(f"Saved failures.csv, gallery.jpg and worst_*.jpg to {out.resolve()}")


if __name__ == "__main__":
    main()
