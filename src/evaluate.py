"""Score trained runs on one split; one row per run, then mean ± SD per group.

Each run is GROUP=path/to/best.pt; the seed is read from the run name (..._s<seed>).
Each model is evaluated at the image size it was trained at. --conf/--iou default to the
Ultralytics evaluation defaults (0.001 / 0.7); pass the locked values for the tuned baseline.

Example:
    python src/evaluate.py --data data/teeth_yolo/data.yaml --split test --out results/baseline_test.csv \
        --runs baseline=runs/baseline_s0/weights/best.pt baseline=runs/baseline_s1/weights/best.pt
"""
import argparse
import csv
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import VAL_CONF, VAL_IOU, seed_of, train_imgsz  # noqa: E402

METRICS = {"mask_mAP50": "map50", "mask_mAP50_95": "map", "mask_P": "mp", "mask_R": "mr"}


def evaluate(weights, data, split="test", conf=VAL_CONF, iou=VAL_IOU, imgsz=None):
    from ultralytics import YOLO
    import tempfile
    m = YOLO(weights).val(data=data, split=split, imgsz=imgsz or train_imgsz(weights), batch=8,
                          conf=conf, iou=iou, plots=False, verbose=False, workers=0,
                          project=str(Path(tempfile.gettempdir()) / "tooth_eval"), name="val", exist_ok=True)
    return m


def row_from(m, group, weights, conf, iou):
    row = {"group": group, "seed": seed_of(weights), "weights": str(weights), "conf": conf, "iou": iou}
    row.update({k: round(float(getattr(m.seg, a)), 4) for k, a in METRICS.items()})
    p, r = row["mask_P"], row["mask_R"]
    row["mask_F1"] = round(2 * p * r / (p + r), 4) if p + r else 0.0
    return row


def summary(rows, keys=("mask_mAP50", "mask_mAP50_95")):
    lines = ["| Group | Seeds | " + " | ".join(keys) + " |", "| --- | --- | " + " | ".join("---" for _ in keys) + " |"]
    for g in dict.fromkeys(r["group"] for r in rows):
        rs = [r for r in rows if r["group"] == g]
        cells = []
        for k in keys:
            v = [r[k] for r in rs]
            sd = st.stdev(v) if len(v) > 1 else 0.0
            cells.append(f"{st.mean(v):.4f} ± {sd:.4f}")
        lines.append(f"| {g} | {', '.join(str(r['seed']) for r in rs)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="GROUP=weights.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=VAL_CONF)
    ap.add_argument("--iou", type=float, default=VAL_IOU)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for spec in args.runs:
        group, weights = spec.split("=", 1)
        rows.append(row_from(evaluate(weights, args.data, args.split, args.conf, args.iou), group, weights,
                             args.conf, args.iou))
        print(rows[-1])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    table = summary(rows)
    out.with_suffix(".md").write_text(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
