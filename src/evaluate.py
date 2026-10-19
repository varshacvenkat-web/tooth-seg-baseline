"""Score trained runs on the test split and summarize across seeds.

Each run is given as GROUP=path/to/best.pt. Writes one row per run plus a
mean, sample standard deviation and min-max range per group, so a change
can be compared against the seed-to-seed noise.

Example:
    python src/evaluate.py --data data/teeth_yolo/data.yaml \
        --runs baseline=runs/baseline_s0/weights/best.pt baseline=runs/baseline_s1/weights/best.pt \
        --out results/baseline.csv
"""
import argparse
import csv
import statistics as st
from pathlib import Path

from ultralytics import YOLO

METRICS = {"mask_mAP50": "map50", "mask_mAP50_95": "map", "mask_P": "mp", "mask_R": "mr"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml whose test split is scored")
    ap.add_argument("--runs", nargs="+", required=True, help="GROUP=weights.pt")
    ap.add_argument("--out", default="results/test_metrics.csv")
    args = ap.parse_args()

    rows = []
    for spec in args.runs:
        group, weights = spec.split("=", 1)
        m = YOLO(weights).val(data=args.data, split="test", imgsz=640, batch=16,
                              plots=False, verbose=False)
        row = {"group": group, "weights": weights}
        row.update({k: round(getattr(m.seg, attr), 4) for k, attr in METRICS.items()})
        rows.append(row)
        print(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n| Group | Runs | Mask mAP@50 | Mask mAP@50:95 |")
    print("| --- | --- | --- | --- |")
    for group in dict.fromkeys(r["group"] for r in rows):
        g = [r for r in rows if r["group"] == group]
        cells = []
        for k in ("mask_mAP50", "mask_mAP50_95"):
            vals = [r[k] for r in g]
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{st.mean(vals):.3f} ± {sd:.3f} (range {min(vals):.3f}–{max(vals):.3f})")
        print(f"| {group} | {len(g)} | {cells[0]} | {cells[1]} |")


if __name__ == "__main__":
    main()
