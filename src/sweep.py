"""Post-processing sweep on the VALIDATION split (no retraining): confidence x NMS IoU.

For every baseline seed and every (conf, iou) pair it records
  - Ultralytics mask mAP@50 and mAP@50:95
  - our own counts at that operating point: precision, recall, F1, duplicate masks per image,
    phantom masks per image and missed teeth per image (src/imgeval.py)
Locking rule (written to configs/postprocess.yaml):
  NMS IoU  = highest mean val mask mAP@50:95 at the lowest confidence in the grid
  conf     = highest mean val F1 at that NMS IoU (the operating point predict.py uses)
It also saves side-by-side examples: loose NMS (highest IoU) vs strict NMS (lowest IoU).

Example:
    python src/sweep.py --data data/teeth_yolo/data.yaml --data-root data/teeth_yolo \
        --runs runs/baseline_s0/weights/best.pt runs/baseline_s1/weights/best.pt --out results/sweep
"""
import argparse
import csv
import os
import statistics as st
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import read_names, seed_of, train_imgsz  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.imgeval import evaluate_frame, gt_in_frame, label_for, predict_frame, split_images  # noqa: E402
from src.common import letterbox, load_image  # noqa: E402


def draw(lb, gm, gt_rec, pm, pred_rec, title):
    out = lb.copy()
    rng = np.random.default_rng(0)
    for m in pm:
        out[m] = (0.55 * out[m] + 0.45 * rng.integers(60, 255, 3)).astype(np.uint8)
    for m, p in zip(pm, pred_rec):
        if p["kind"] == "duplicate":
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, (0, 0, 255), 2)
    for m, g in zip(gm, gt_rec):
        if not g["ok"]:
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, (0, 220, 255), 2)
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--data-root", required=True, help="YOLO dataset folder (images/val, labels/val)")
    ap.add_argument("--runs", nargs="+", required=True, help="baseline best.pt files, one per seed")
    ap.add_argument("--confs", default="0.001,0.25,0.5,0.75")
    ap.add_argument("--ious", default="0.4,0.5,0.6,0.7")
    ap.add_argument("--out", default="results/sweep")
    ap.add_argument("--lock-out", default=os.environ.get("POSTPROCESS_YAML", "configs/postprocess.yaml"))
    ap.add_argument("--n-visuals", type=int, default=4)
    args = ap.parse_args()

    from ultralytics import YOLO
    confs = [float(x) for x in args.confs.split(",")]
    ious = [float(x) for x in args.ious.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = read_names(args.data)
    images = split_images(args.data_root, "val")

    rows = []
    for w in args.runs:
        seed, imgsz = seed_of(w), train_imgsz(w)
        model = YOLO(w)
        acc = {(c, i): {"tp": 0, "pred": 0, "gt": 0, "dup": 0, "phantom": 0, "fn": 0} for c in confs for i in ious}
        for p in images:
            lb, meta = letterbox(load_image(p), imgsz)
            gm, gn = gt_in_frame(label_for(p), names, meta)
            for c in confs:
                for i in ious:
                    pm, pc, pn = predict_frame(model, lb, c, i, imgsz)
                    g, pr = evaluate_frame(gm, gn, pm, pc, pn)
                    a = acc[(c, i)]
                    a["gt"] += len(g)
                    a["pred"] += len(pr)
                    a["tp"] += sum(x["kind"] == "tp" for x in pr)
                    a["dup"] += sum(x["kind"] == "duplicate" for x in pr)
                    a["phantom"] += sum(x["kind"] == "phantom" for x in pr)
                    a["fn"] += sum(not x["ok"] for x in g)
        for c in confs:
            for i in ious:
                m = evaluate(w, args.data, "val", c, i, imgsz)
                a, n = acc[(c, i)], len(images)
                P = a["tp"] / a["pred"] if a["pred"] else 0.0
                R = a["tp"] / a["gt"] if a["gt"] else 0.0
                rows.append({"seed": seed, "conf": c, "iou": i,
                             "mask_mAP50": round(float(m.seg.map50), 4), "mask_mAP50_95": round(float(m.seg.map), 4),
                             "precision": round(P, 4), "recall": round(R, 4),
                             "f1": round(2 * P * R / (P + R), 4) if P + R else 0.0,
                             "dup_per_img": round(a["dup"] / n, 3), "phantom_per_img": round(a["phantom"] / n, 3),
                             "missed_per_img": round(a["fn"] / n, 3), "pred_per_img": round(a["pred"] / n, 2)})
                print(rows[-1])

    with (out / "sweep_val.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    def agg(c, i, k):
        v = [r[k] for r in rows if r["conf"] == c and r["iou"] == i]
        return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)

    keys = ["mask_mAP50", "mask_mAP50_95", "precision", "recall", "f1", "dup_per_img", "missed_per_img", "phantom_per_img"]
    lines = ["| conf | NMS IoU | " + " | ".join(keys) + " |", "| --- | --- | " + " | ".join("---" for _ in keys) + " |"]
    for c in confs:
        for i in ious:
            lines.append(f"| {c} | {i} | " + " | ".join(
                f"{agg(c, i, k)[0]:.3f} ± {agg(c, i, k)[1]:.3f}" for k in keys) + " |")

    iou_star = max(ious, key=lambda i: agg(min(confs), i, "mask_mAP50_95")[0])
    operating = [c for c in confs if c >= 0.05] or confs   # 0.001 is the mAP reference, not an operating point
    conf_star = max(operating, key=lambda c: agg(c, iou_star, "f1")[0])
    lock = {"conf": conf_star, "iou": iou_star, "split": "val", "seeds": [seed_of(w) for w in args.runs],
            "rule": "iou = best mean val mask mAP50-95 at the lowest conf; conf = best mean val F1 at that iou",
            "val_mask_mAP50_95": round(agg(min(confs), iou_star, "mask_mAP50_95")[0], 4),
            "val_f1": round(agg(conf_star, iou_star, "f1")[0], 4)}
    Path(args.lock_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.lock_out).write_text(yaml.safe_dump(lock, sort_keys=False))
    lines += ["", f"**Locked on val:** NMS IoU = {iou_star}, confidence = {conf_star} ({lock['rule']})."]
    (out / "sweep_val.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # visual examples with the first seed: loose vs strict NMS at the locked confidence
    model, w = YOLO(args.runs[0]), args.runs[0]
    imgsz, loose, strict = train_imgsz(w), max(ious), min(ious)
    scored = []
    for p in images:
        lb, meta = letterbox(load_image(p), imgsz)
        gm, gn = gt_in_frame(label_for(p), names, meta)
        res = {}
        for tag, i in (("loose", loose), ("strict", strict)):
            pm, pc, pn = predict_frame(model, lb, conf_star, i, imgsz)
            g, pr = evaluate_frame(gm, gn, pm, pc, pn)
            res[tag] = (pm, g, pr)
        d = sum(x["kind"] == "duplicate" for x in res["loose"][2])
        mi = sum(not x["ok"] for x in res["strict"][1]) - sum(not x["ok"] for x in res["loose"][1])
        scored.append((d + max(mi, 0), p, lb, gm, res))
    scored.sort(key=lambda t: -t[0])
    for k, (_, p, lb, gm, res) in enumerate(scored[: args.n_visuals], 1):
        panels = []
        for tag, i in (("loose", loose), ("strict", strict)):
            pm, g, pr = res[tag]
            nd, nm = sum(x["kind"] == "duplicate" for x in pr), sum(not x["ok"] for x in g)
            panels.append(draw(lb, gm, g, pm, pr, f"{p.name} NMS IoU {i}: {nd} duplicates (red), {nm} missed (yellow)"))
        cv2.imwrite(str(out / f"nms_example_{k:02d}_{p.stem}.jpg"), np.hstack(panels))
    print(f"Saved sweep_val.csv, sweep_val.md, nms_example_*.jpg to {out.resolve()} and locked {args.lock_out}")


if __name__ == "__main__":
    main()
