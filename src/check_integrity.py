"""Split integrity (leakage) check and baseline configuration audit.

Leakage: the dataset export has no patient IDs, so patient-level grouping is impossible to
verify directly. As the closest check we confirm that
  1. no file name appears in more than one split,
  2. no byte-identical image appears in more than one split (MD5),
  3. no near-duplicate image (64-bit difference hash, Hamming distance <= 4) crosses splits,
     which would catch the same radiograph re-exported or slightly re-cropped.
Config audit: reads each run's args.yaml and states input resolution, normalization, NMS,
optimizer, epochs and which split chose the checkpoint (it must be val, never test).

Example:
    python src/check_integrity.py --data-root data/teeth_yolo --runs runs/baseline_s0 runs/baseline_s1 \
        --out results/integrity.md
"""
import argparse
import csv
import hashlib
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PAD_VALUE  # noqa: E402

SPLITS = ("train", "val", "test")


def dhash(path, size=8) -> int:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def check_splits(root):
    files = {s: sorted((Path(root) / "images" / s).iterdir()) for s in SPLITS}
    L, ok = [], True
    names = {s: {p.name for p in v} for s, v in files.items()}
    for a, b in itertools.combinations(SPLITS, 2):
        both = names[a] & names[b]
        ok &= not both
        L.append(f"- File names shared by {a} and {b}: {len(both)}" + (f" ({sorted(both)[:5]})" if both else ""))
    md5 = {}
    for s, v in files.items():
        for p in v:
            md5.setdefault(hashlib.md5(p.read_bytes()).hexdigest(), []).append((s, p.name))
    cross = [v for v in md5.values() if len({s for s, _ in v}) > 1]
    within = [v for v in md5.values() if len(v) > 1 and len({s for s, _ in v}) == 1]
    ok &= not cross
    L.append(f"- Byte-identical images across splits: {len(cross)}" + (f" {cross[:3]}" if cross else ""))
    L.append(f"- Byte-identical images within one split (no leakage, noted): {len(within)}")
    hashes = [(s, p.name, dhash(p)) for s, v in files.items() for p in v]
    near = []
    for (s1, n1, h1), (s2, n2, h2) in itertools.combinations(hashes, 2):
        if s1 != s2 and bin(h1 ^ h2).count("1") <= 4:
            near.append((s1, n1, s2, n2, bin(h1 ^ h2).count("1")))
    L.append(f"- Near-duplicate pairs across splits (dHash Hamming <= 4): {len(near)}")
    for n in near[:10]:
        L.append(f"    - {n[0]}/{n[1]} ~ {n[2]}/{n[3]} (distance {n[4]}): inspect by eye")
    counts = ", ".join(f"{s} {len(v)}" for s, v in files.items())
    return ok and not near, L, counts, near


def audit(run):
    run = Path(run)
    a = yaml.safe_load((run / "args.yaml").read_text())
    lr = None
    res = run / "results.csv"
    if res.exists():
        rows = list(csv.DictReader(res.open()))
        col = [k for k in rows[0] if k.strip() == "lr/pg0"]
        if col:
            lr = max(float(r[col[0]]) for r in rows)
    return a, lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--runs", nargs="*", default=[], help="run folders (with args.yaml)")
    ap.add_argument("--out", default="results/integrity.md")
    args = ap.parse_args()

    ok, lines, counts, near = check_splits(args.data_root)
    L = ["## Split integrity", "", f"Split sizes: {counts}.", ""] + lines + ["",
         ("**No leakage found**" if ok else "**Possible leakage: see above**") +
         " at the file and image level. The Kaggle export carries no patient identifiers, so images of the same "
         "patient taken on different visits cannot be grouped and could not be checked; this is a stated "
         "limitation. Cross-source generalisation is untested: all images come from one dataset; a second "
         "source (e.g. the Tufts Dental Database, by access request) would be needed."]

    if args.runs:
        L += ["", "## Baseline configuration audit", "",
              "| Setting | " + " | ".join(Path(r).name for r in args.runs) + " |",
              "| --- | " + " | ".join("---" for _ in args.runs) + " |"]
        audits = [audit(r) for r in args.runs]
        keys = ["model", "imgsz", "epochs", "batch", "optimizer", "lr0", "seed", "deterministic", "iou", "conf",
                "split", "fliplr", "mosaic", "close_mosaic", "hsv_v", "scale", "amp", "data"]
        for k in keys:
            L.append(f"| {k} | " + " | ".join(str(a.get(k)) for a, _ in audits) + " |")
        L.append("| peak learning rate used (results.csv) | " + " | ".join(
            f"{lr:.6f}" if lr else "n/a" for _, lr in audits) + " |")
        splits = {str(a.get("split")) for a, _ in audits}
        L += ["", f"- Checkpoint selection split: {', '.join(sorted(splits))}"
              + (" (validation only: the test split never chose a checkpoint)." if splits == {"val"} else
                 " **WARNING: not the validation split.**"),
              "- Input: long side resized to imgsz, aspect kept, padded with gray "
              f"{PAD_VALUE}; pixels scaled to 0-1, BGR->RGB (Ultralytics; predict.py uses the same, see tests).",
              "- Anchors: none. YOLO11 heads are anchor-free, so there are no anchor settings to tune.",
              "- NMS IoU during validation = the iou row above (0.7 default); confidence 0.001 for mAP.",
              "- optimizer 'auto' picks AdamW for short runs and ignores lr0; the peak rate actually used is shown."]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
