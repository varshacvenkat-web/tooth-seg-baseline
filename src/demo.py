"""predict.py demo on 10 images the model never trained on, plus every edge case.

Inputs: --images (a folder of new X-rays) if given; otherwise 10 TEST-split images (never used for
training or checkpoint selection), including the 2 hardest metal/crown and overlapping-teeth cases
from the failure analysis. Writes overlays, a contact sheet, a confidence-threshold count table and
an edge-case log.

Example:
    python src/demo.py --weights runs/baseline_s0/weights/best.pt --data-root data/teeth_yolo --out results/demo
"""
import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.imgeval import split_images  # noqa: E402


def pick(data_root, failures_csv, n=10, seed=0):
    test = split_images(data_root, "test")
    hard = []
    if Path(failures_csv).exists():
        rows = list(csv.DictReader(open(failures_csv)))
        for col in ("metallic / restoration", "merged neighbours"):
            r = max(rows, key=lambda r: int(r.get(col, 0) or 0), default=None)
            if r and int(r.get(col, 0) or 0) > 0 and r["image"] not in hard:
                hard.append(r["image"])
    chosen = [p for p in test if p.name in hard]
    rest = [p for p in test if p.name not in hard]
    random.Random(seed).shuffle(rest)
    return chosen + rest[: n - len(chosen)], [p.name for p in chosen]


def edge_cases(tmp, src_img):
    tmp.mkdir(parents=True, exist_ok=True)
    g = cv2.imread(str(src_img), cv2.IMREAD_GRAYSCALE)
    cases = {"corrupted.jpg": None, "tiny_20px.png": np.zeros((20, 20), np.uint8),
             "colour_tinted.png": cv2.merge([g, (g * 0.8).astype(np.uint8), np.full_like(g, 90)]),
             "sixteen_bit.png": g.astype(np.uint16) * 257,
             "rgba.png": cv2.cvtColor(g, cv2.COLOR_GRAY2BGRA),
             "blank_no_teeth.png": np.full_like(g, 30)}
    for name, arr in cases.items():
        if arr is None:
            (tmp / name).write_bytes(b"\xff\xd8 this is not a real jpeg")
        else:
            cv2.imwrite(str(tmp / name), arr)
    return tmp


def run_predict(src, weights, out, extra=()):
    cmd = [sys.executable, str(ROOT / "predict.py"), str(src), "--weights", str(weights), "--out", str(out), *extra]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--images", help="folder of new X-rays (overrides the test-split pick)")
    ap.add_argument("--failures-csv", default="results/failures/failures_worst20.csv")
    ap.add_argument("--out", default="results/demo")
    args = ap.parse_args()

    out = Path(args.out)
    inputs = out / "inputs"
    if out.exists():
        shutil.rmtree(out)
    inputs.mkdir(parents=True)
    if args.images:
        imgs, hard, source = sorted(Path(args.images).iterdir()), [], f"new images from {args.images}"
    else:
        imgs, hard = pick(args.data_root, args.failures_csv)
        source = "test-split images (never trained on or used for checkpoint selection)"
    for p in imgs:
        shutil.copy(p, inputs / p.name)

    code, log = run_predict(inputs, args.weights, out / "predictions")
    _, counts_log = run_predict(inputs, args.weights, out / "conf_counts", ["--count-sweep", "0.1,0.25,0.5,0.75"])

    tiles = []
    for p in sorted(inputs.iterdir()):
        ov = out / "predictions" / f"{p.stem}_overlay.jpg"
        if not ov.exists():
            continue
        a, b = cv2.imread(str(p)), cv2.imread(str(ov))
        pair = np.hstack([cv2.resize(a, (600, int(a.shape[0] * 600 / a.shape[1]))),
                          cv2.resize(b, (600, int(b.shape[0] * 600 / b.shape[1])))])
        bar = np.zeros((22, pair.shape[1], 3), np.uint8)
        n = json.loads((out / "predictions" / f"{p.stem}.json").read_text())["num_teeth"]
        tag = " (hard case)" if p.name in hard else ""
        cv2.putText(bar, f"{p.name}{tag}: input (left), {n} teeth predicted (right)", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(np.vstack([bar, pair]))
    if tiles:
        w = max(t.shape[1] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, 4, 0, w - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
        cv2.imwrite(str(out / "demo_contact_sheet.jpg"), np.vstack(tiles))

    ec = edge_cases(out / "edge_cases", imgs[0])
    _, ec_log = run_predict(ec, args.weights, out / "edge_case_outputs")
    _, missing_log = run_predict(ec / "does_not_exist.png", args.weights, out / "edge_case_outputs")

    L = ["## predict.py demo", "", f"Inputs: {len(imgs)} {source}." +
         (f" Hard cases: {', '.join(hard)}." if hard else ""), "", "```", log, "```", "",
         "### Teeth per image at different confidence thresholds", "", counts_log.split("\n\n")[-1] if counts_log else "",
         "", "### Edge cases", "", "```", ec_log, missing_log, "```", "",
         "Outputs per image: JSON (tooth number, confidence, box, polygon), label PNG and N x H x W binary masks "
         "at the original resolution, overlay JPG drawn from the polygons (alignment check)."]
    (out / "demo.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
