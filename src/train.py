"""Train YOLO11s-seg with the locked baseline settings (src/common.py: TRAIN_SETTINGS).

Only --seed changes between baseline runs. An experiment changes exactly one setting with
--set (e.g. --set imgsz=1024); the change is recorded in <run>/change.json.
A run that already finished (all epochs in results.csv) is skipped, so reruns are cheap.

Examples:
    python src/train.py --data data/teeth_yolo/data.yaml --seed 1 --name baseline_s1
    python src/train.py --data data/teeth_yolo/data.yaml --seed 1 --name exp1_s1 --set imgsz=1024
"""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import TRAIN_SETTINGS  # noqa: E402


def parse_value(v: str):
    return yaml.safe_load(v)  # "1024" -> 1024, "0.0" -> 0.0, "false" -> False


def finished(run: Path, epochs: int) -> bool:
    res = run / "results.csv"
    if not (run / "weights" / "best.pt").exists() or not res.exists():
        return False
    return len(res.read_text().strip().splitlines()) - 1 >= epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default=os.environ.get("RUNS_DIR", "runs"))
    ap.add_argument("--model", default=os.environ.get("BASE_MODEL", "yolo11s-seg.pt"))
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides (the experiment's one change)")
    args = ap.parse_args()

    change = dict(kv.split("=", 1) for kv in args.set)
    change = {k: parse_value(v) for k, v in change.items()}
    # EXTRA_SET is only for quick smoke tests (e.g. "epochs=2 imgsz=320"); never for reported runs
    extra = {k: parse_value(v) for k, v in (kv.split("=", 1) for kv in os.environ.get("EXTRA_SET", "").split())}
    settings = {**TRAIN_SETTINGS, **extra, **change}

    project = Path(args.project).resolve()  # absolute: newer Ultralytics nests relative paths
    run = project / args.name
    if finished(run, settings["epochs"]):
        print(f"{args.name}: already trained ({run}), skipping")
        return
    from ultralytics import YOLO
    YOLO(args.model).train(data=args.data, seed=args.seed, project=str(project), name=args.name,
                           exist_ok=True, **settings)
    (run / "change.json").write_text(json.dumps({"seed": args.seed, "change": change, "settings": settings}, indent=1))


if __name__ == "__main__":
    main()
