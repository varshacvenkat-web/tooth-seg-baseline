"""Split integrity (leakage) check and baseline configuration audit.

Leakage: the dataset export has no patient IDs, so patient-level grouping is impossible to
verify directly. As the closest check we confirm that
  1. no file name appears in more than one split,
  2. no byte-identical image appears in more than one split (MD5),
  3. no near-duplicate image crosses splits. Two stages, because every panoramic X-ray shares the
     same overall layout and a coarse fingerprint alone flags many different patients:
       a. candidates: 64-bit difference hash, Hamming distance <= 6
       b. confirmed:  pixel correlation >= 0.95 between the two images at 256 x 128
     The closest candidate pairs are saved side by side (near_duplicates.jpg) to check by eye.
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


def check_splits(root, out_dir=None):
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
    hashes = [(s, p, dhash(p)) for s, v in files.items() for p in v]
    cand = [(s1, p1, s2, p2, bin(h1 ^ h2).count("1"))
            for (s1, p1, h1), (s2, p2, h2) in itertools.combinations(hashes, 2)
            if s1 != s2 and bin(h1 ^ h2).count("1") <= 6]
    small = {}

    def thumb(p):
        if p not in small:
            g = cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (256, 128), interpolation=cv2.INTER_AREA)
            g = g.astype(np.float32).ravel()
            small[p] = (g - g.mean()) / (g.std() + 1e-6)
        return small[p]

    scored = sorted(((float(np.mean(thumb(a) * thumb(b))), s1, a, s2, b, d) for s1, a, s2, b, d in cand),
                    key=lambda t: -t[0])
    near = [t for t in scored if t[0] >= 0.95]
    corr = [t[0] for t in scored]
    L.append(f"- Near-duplicate candidates across splits (dHash Hamming <= 6): {len(cand)}"
             + (f"; their pixel correlation ranges {min(corr):.2f} to {max(corr):.2f}, median {np.median(corr):.2f}" if corr else ""))
    L.append(f"- Confirmed near-duplicates across splits (correlation >= 0.95): {len(near)}")
    for c, s1, a, s2, b, d in near[:10]:
        L.append(f"    - {s1}/{a.name} ~ {s2}/{b.name} (correlation {c:.3f})")
    if out_dir is not None and scored:
        tiles = []
        for c, s1, a, s2, b, d in scored[:12]:
            ia = cv2.resize(cv2.imread(str(a)), (480, 240))
            ib = cv2.resize(cv2.imread(str(b)), (480, 240))
            bar = np.zeros((22, 964, 3), np.uint8)
            cv2.putText(bar, f"{s1}/{a.name}  vs  {s2}/{b.name}   correlation {c:.3f}", (6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(np.vstack([bar, np.hstack([ia, np.full((240, 4, 3), 255, np.uint8), ib])]))
        cv2.imwrite(str(Path(out_dir) / "near_duplicates.jpg"), np.vstack(tiles))
        L.append("- The 12 closest candidate pairs are in near_duplicates.jpg: same patient = same teeth, "
                 "fillings and gaps; different patients only share the general layout.")
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

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ok, lines, counts, near = check_splits(args.data_root, Path(args.out).parent)
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
