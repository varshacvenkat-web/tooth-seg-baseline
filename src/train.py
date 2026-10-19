"""Train the baseline: YOLO11s-seg, Ultralytics defaults except epochs=50.

Only --seed (and --data / --name) should change between runs, so any
difference in results comes from the change being tested.

Example:
    python src/train.py --data data/teeth_yolo/data.yaml --seed 1 --name baseline_s1
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

SETTINGS = dict(epochs=50, imgsz=640, batch=16, deterministic=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--model", default="yolo11s-seg.pt")
    args = ap.parse_args()

    # absolute path: newer Ultralytics nests a relative project under runs/segment/
    project = str(Path(args.project).resolve())
    YOLO(args.model).train(data=args.data, seed=args.seed, project=project,
                           name=args.name, exist_ok=True, **SETTINGS)


if __name__ == "__main__":
    main()
