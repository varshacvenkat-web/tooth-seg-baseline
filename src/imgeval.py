"""Per-image, per-tooth evaluation in the model's input frame (IMGSZ x IMGSZ letterbox).

Ground-truth polygons are drawn into the same letterboxed frame the model sees, so predicted
and true masks are compared pixel for pixel without any resizing. Used by sweep.py,
analyze.py and failures.py so all three count teeth the same way.

Matching (greedy by confidence, like COCO): each prediction takes the unmatched true tooth
it overlaps most; IoU >= 0.5 and the same tooth number = true positive.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.common import IMGSZ, letterbox, load_image, mirror, neighbor, tooth_number

MATCH_IOU = 0.5


def gt_in_frame(label_file, names, meta):
    """YOLO labels -> (k x S x S bool masks, tooth numbers), merged per tooth number."""
    S = meta["imgsz"]
    (h, w), (top, left) = meta["resized_hw"], meta["pad_tl"]
    by_tooth = {}
    for line in Path(label_file).read_text().splitlines():
        v = line.split()
        if len(v) < 7:
            continue
        pts = np.array(v[1:], float).reshape(-1, 2) * [w, h] + [left, top]
        n = tooth_number(names[int(v[0])])
        m = by_tooth.setdefault(n, np.zeros((S, S), np.uint8))
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    nums = sorted(by_tooth)
    masks = np.stack([by_tooth[n].astype(bool) for n in nums]) if nums else np.zeros((0, S, S), bool)
    return masks, nums


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    A = a.reshape(len(a), -1).astype(np.float32)
    B = b.reshape(len(b), -1).astype(np.float32)
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def predict_frame(model, lb, conf, iou, imgsz=IMGSZ):
    r = model.predict(lb, imgsz=imgsz, conf=conf, iou=iou, retina_masks=True, verbose=False)[0]
    if r.masks is None or len(r.boxes) == 0:
        return np.zeros((0, imgsz, imgsz), bool), np.zeros(0), []
    masks = r.masks.data.cpu().numpy() > 0.5
    confs = r.boxes.conf.cpu().numpy()
    nums = [tooth_number(model.names[int(c)]) for c in r.boxes.cls.cpu().numpy()]
    order = np.argsort(-confs)
    return masks[order], confs[order], [nums[i] for i in order]


def evaluate_frame(gt_masks, gt_nums, pm, pc, pn, img_lb=None, meta=None):
    """Score one image. Returns (per-true-tooth records, per-prediction records)."""
    G = iou_matrix(gt_masks, pm)            # true x pred
    P = iou_matrix(pm, pm)                  # pred x pred
    matched_gt, pred_rec = set(), []
    for j in range(len(pm)):
        cand = [(G[i, j], i) for i in range(len(gt_nums)) if i not in matched_gt and gt_nums[i] == pn[j]]
        best = max(cand, default=(0.0, None))
        dup = bool(j and P[j, :j].max() > MATCH_IOU)   # overlaps a more confident prediction
        any_overlap = float(G[:, j].max()) if len(gt_nums) else 0.0
        if best[0] >= MATCH_IOU:
            matched_gt.add(best[1])
            kind = "tp"
        elif dup:
            kind = "duplicate"
        elif any_overlap >= MATCH_IOU:
            kind = "wrong_number"
        elif any_overlap > 0.1:
            kind = "poor_boundary"
        else:
            kind = "phantom"                   # on bone, sinus, background: no tooth there
        pred_rec.append({"tooth": pn[j], "conf": float(pc[j]), "kind": kind, "iou_gt": any_overlap})

    # bright-metal and field-of-view helpers
    bright = content = None
    if img_lb is not None and meta is not None:
        gray = cv2.cvtColor(img_lb, cv2.COLOR_BGR2GRAY)
        (h, w), (top, left) = meta["resized_hw"], meta["pad_tl"]
        content = (top, left, top + h, left + w)
        inside = gray[top:top + h, left:left + w]
        bright = gray >= max(200, np.percentile(inside, 99.5))

    # coverage[i, j] = share of true tooth i covered by prediction j
    if len(gt_nums) and len(pm):
        A = gt_masks.reshape(len(gt_nums), -1).astype(np.float32)
        cov = (A @ pm.reshape(len(pm), -1).astype(np.float32).T) / np.maximum(A.sum(1)[:, None], 1)
        big = cov >= 0.3
        merged_flags = [bool((big[i] & (big.sum(0) >= 2)).any()) for i in range(len(gt_nums))]
    else:
        merged_flags = [False] * len(gt_nums)

    gt_rec = []
    for i, n in enumerate(gt_nums):
        same = [G[i, j] for j in range(len(pm)) if pn[j] == n]
        best_same = float(max(same, default=0.0))
        j_any = int(np.argmax(G[i])) if len(pm) else None
        best_any = float(G[i, j_any]) if j_any is not None else 0.0
        area = int(gt_masks[i].sum())
        # merged: one prediction covers >= 30% of this tooth and >= 30% of another tooth
        merged = merged_flags[i]
        rec = {"tooth": n, "iou": best_same, "best_any": best_any, "area": area,
               "pred_as": pn[j_any] if j_any is not None and best_any > 0 else None, "merged": merged,
               "metal": False, "edge": False}
        if bright is not None:
            rec["metal"] = bool(bright[gt_masks[i]].mean() >= 0.15) if area else False
            ys, xs = np.nonzero(gt_masks[i])
            if len(xs):
                t, l, b, r = content
                rec["edge"] = bool(xs.min() <= l + 2 or xs.max() >= r - 3 or ys.min() <= t + 2 or ys.max() >= b - 3)
        rec["ok"] = i in matched_gt
        rec["category"] = categorize(rec)
        gt_rec.append(rec)
    return gt_rec, pred_rec


def categorize(r) -> str:
    """One primary failure category per true tooth (cause-specific ones first)."""
    if r["ok"]:
        return "ok"
    if r["edge"]:
        return "edge clipping"
    if r["metal"]:
        return "metallic / restoration"
    if r["merged"]:
        return "merged neighbours"
    if r["best_any"] < 0.3:
        return "missed"
    if r["best_any"] >= 0.5 and r["pred_as"] is not None and r["pred_as"] != r["tooth"]:
        if neighbor(r["pred_as"], r["tooth"]):   # first: central incisors 8/9, 24/25 are also mirrors
            return "wrong number (neighbour)"
        if r["pred_as"] == mirror(r["tooth"]):
            return "wrong number (mirror)"
        return "wrong number (other)"
    return "poor boundary"


def image_score(gt_rec) -> float:
    """Per-image mean mask IoU: each true tooth's best same-number IoU, missed = 0."""
    return float(np.mean([r["iou"] for r in gt_rec])) if gt_rec else 1.0


def run_image(model, img_path, label_path, names, conf, iou, imgsz=IMGSZ):
    img = load_image(img_path)
    lb, meta = letterbox(img, imgsz)
    gm, gn = gt_in_frame(label_path, names, meta)
    pm, pc, pn = predict_frame(model, lb, conf, iou, imgsz)
    gt_rec, pred_rec = evaluate_frame(gm, gn, pm, pc, pn, lb, meta)
    return {"image": Path(img_path).name, "score": image_score(gt_rec), "gt": gt_rec, "pred": pred_rec,
            "lb": lb, "meta": meta, "gm": gm, "gn": gn, "pm": pm, "pn": pn, "pc": pc}


def split_images(data_root, split):
    d = Path(data_root) / "images" / split
    return sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def label_for(img_path):
    p = Path(img_path)
    return p.parents[2] / "labels" / p.parent.name / f"{p.stem}.txt"


def counts(results):
    """Totals over images: TP/FP/FN and their breakdown."""
    c = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}
    fp_kinds, fn_kinds = {}, {}
    for r in results:
        c["gt"] += len(r["gt"])
        c["pred"] += len(r["pred"])
        for p in r["pred"]:
            if p["kind"] == "tp":
                c["tp"] += 1
            else:
                c["fp"] += 1
                fp_kinds[p["kind"]] = fp_kinds.get(p["kind"], 0) + 1
        for g in r["gt"]:
            if not g["ok"]:
                c["fn"] += 1
                fn_kinds[g["category"]] = fn_kinds.get(g["category"], 0) + 1
    c["precision"] = c["tp"] / c["pred"] if c["pred"] else 0.0
    c["recall"] = c["tp"] / c["gt"] if c["gt"] else 0.0
    pr = c["precision"] + c["recall"]
    c["f1"] = 2 * c["precision"] * c["recall"] / pr if pr else 0.0
    return c, fp_kinds, fn_kinds
