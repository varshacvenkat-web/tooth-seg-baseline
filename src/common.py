"""Shared settings and helpers: one source of truth for training, evaluation and predict.py.

Preprocessing here mirrors what Ultralytics does when it evaluates (see tests/test_preprocess.py,
which fails if the two ever diverge):
  1. read as 8-bit, 3-channel BGR (grayscale X-rays are replicated to 3 channels)
  2. resize the long side to IMGSZ with bilinear interpolation, keeping the aspect ratio
  3. pad to IMGSZ x IMGSZ with gray (114), image centered
The model itself converts BGR->RGB and scales pixels to 0-1 internally.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- settings
IMGSZ = 640
PAD_VALUE = 114
SEED_SPLIT = 42
TRAIN_SETTINGS = dict(epochs=50, imgsz=IMGSZ, batch=16, deterministic=True)
VAL_CONF = 0.001   # Ultralytics evaluation default: keep (almost) every prediction for mAP
VAL_IOU = 0.7      # Ultralytics NMS IoU default
PREDICT_CONF = 0.25
MIN_AREA_FRAC = 0.0005  # predict.py drops masks smaller than 0.05% of the image

# ---------------------------------------------------------------- teeth
def tooth_number(name) -> int:
    """'14' -> 14, '13(polygon)' -> 13, unknown -> 0."""
    m = re.match(r"\d+", str(name))
    return int(m.group()) if m else 0


def mirror(n: int) -> int:
    """Same tooth on the other side of the jaw (universal numbering)."""
    return 17 - n if n <= 16 else 49 - n


def neighbor(a: int, b: int) -> bool:
    return abs(a - b) == 1 and (a <= 16) == (b <= 16)


TOOTH_TYPES = {
    "incisor": {7, 8, 9, 10, 23, 24, 25, 26},
    "canine": {6, 11, 22, 27},
    "premolar": {4, 5, 12, 13, 20, 21, 28, 29},
    "molar (1st/2nd)": {2, 3, 14, 15, 18, 19, 30, 31},
    "third molar": {1, 16, 17, 32},
}
ROOTS = {"incisor": "single-rooted", "canine": "single-rooted", "premolar": "mostly single-rooted",
         "molar (1st/2nd)": "multi-rooted", "third molar": "multi-rooted"}


def tooth_type(n: int) -> str:
    for t, members in TOOTH_TYPES.items():
        if n in members:
            return t
    return "other"

# ---------------------------------------------------------------- images
def load_image(path) -> np.ndarray:
    """Read any X-ray into the model's input format: H x W x 3, uint8, BGR, gray replicated.

    Raises FileNotFoundError for a missing file and ValueError for anything unreadable
    or unusable, with a message that says what is wrong.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input image not found: {p}")
    # IMREAD_COLOR = exactly how Ultralytics reads training/eval images (8-bit, 3-channel BGR,
    # EXIF rotation applied, alpha dropped, 16-bit scaled down, gray replicated)
    raw = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError(f"Could not decode {p.name}: the file is corrupted or not an image format OpenCV reads")
    return to_model_format(raw, name=p.name)


def to_model_format(img: np.ndarray, name: str = "image") -> np.ndarray:
    if not isinstance(img, np.ndarray):
        raise ValueError(f"{name}: expected a numpy array, got {type(img).__name__}")
    if img.ndim not in (2, 3) or (img.ndim == 3 and img.shape[2] not in (1, 3, 4)):
        raise ValueError(f"{name}: unsupported shape {img.shape}; expected H x W, H x W x 1, x 3 or x 4")
    h, w = img.shape[:2]
    if min(h, w) < 32:
        raise ValueError(f"{name}: image is {w}x{h} px; at least 32 px per side is required")
    if img.dtype == np.uint16:          # same rule as cv2.IMREAD_COLOR: keep the high byte
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]               # drop alpha
    if img.ndim == 3 and img.shape[2] == 3:
        if is_color(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # real colour input: X-rays are grayscale
        else:
            return np.ascontiguousarray(img)  # gray-looking 3-channel: unchanged (training parity)
    if img.ndim == 3:
        img = img[..., 0]
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def is_color(img: np.ndarray, tol: float = 3.0) -> bool:
    """True for genuinely coloured images. Grayscale JPEGs saved as 3 channels differ between
    channels by a level or two from compression; those count as gray and are left untouched."""
    if img.ndim != 3 or img.shape[2] < 3:
        return False
    s = img[::4, ::4, :3].astype(np.int16)
    return float(np.abs(s[..., 0] - s[..., 1]).mean() + np.abs(s[..., 1] - s[..., 2]).mean()) > tol


def letterbox(img: np.ndarray, imgsz: int = IMGSZ):
    """Resize the long side to imgsz (aspect kept, bilinear) and pad to a centered square.

    Returns the padded image and the metadata needed to undo it.
    """
    h0, w0 = img.shape[:2]
    r = imgsz / max(h0, w0)
    if r != 1:
        w, h = min(math.ceil(w0 * r), imgsz), min(math.ceil(h0 * r), imgsz)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    top, left = (imgsz - h) // 2, (imgsz - w) // 2
    out = cv2.copyMakeBorder(img, top, imgsz - h - top, left, imgsz - w - left,
                             cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    meta = {"orig_hw": (h0, w0), "resized_hw": (h, w), "pad_tl": (top, left), "imgsz": imgsz}
    return out, meta


def unletterbox_mask(mask: np.ndarray, meta: dict) -> np.ndarray:
    """Map a mask in letterboxed (imgsz x imgsz) coordinates back to the original image: bool, H0 x W0."""
    (h0, w0), (h, w), (top, left) = meta["orig_hw"], meta["resized_hw"], meta["pad_tl"]
    crop = mask[top:top + h, left:left + w].astype(np.float32)
    return cv2.resize(crop, (w0, h0), interpolation=cv2.INTER_LINEAR) > 0.5

# ---------------------------------------------------------------- masks and IoU
def rasterize(polys, h, w, s=1.0):
    """Polygons in pixel coordinates -> {'m': mask at scale s, 'box': crop box, 'area': pixels}."""
    m = np.zeros((int(h * s) + 2, int(w * s) + 2), np.uint8)
    pts = []
    for p in polys:
        q = np.round(np.asarray(p, float) * s).astype(np.int32)
        if len(q) >= 3:
            cv2.fillPoly(m, [q], 1)
            pts.append(q)
    if not pts:
        return {"m": m, "box": (0, 0, 0, 0), "area": 0}
    q = np.concatenate(pts)
    x0, y0 = q.min(0)
    x1, y1 = q.max(0) + 1
    return {"m": m, "box": (max(int(x0), 0), max(int(y0), 0), int(x1), int(y1)), "area": int(m.sum())}


def inter(a, b) -> int:
    x0, y0 = max(a["box"][0], b["box"][0]), max(a["box"][1], b["box"][1])
    x1, y1 = min(a["box"][2], b["box"][2]), min(a["box"][3], b["box"][3])
    if x1 <= x0 or y1 <= y0:
        return 0
    return int((a["m"][y0:y1, x0:x1] & b["m"][y0:y1, x0:x1]).sum())


def iou(a, b) -> float:
    i = inter(a, b)
    u = a["area"] + b["area"] - i
    return i / u if u else 0.0


def load_gt(label_file, names, w, h):
    """YOLO polygon labels -> {tooth number: [polygons in pixels]}."""
    gt = {}
    for line in Path(label_file).read_text().splitlines():
        v = line.split()
        if len(v) < 7:
            continue
        pts = np.array(v[1:], float).reshape(-1, 2) * [w, h]
        gt.setdefault(tooth_number(names[int(v[0])]), []).append(pts)
    return gt

# ---------------------------------------------------------------- datasets
def read_names(data_yaml) -> dict:
    import yaml
    d = yaml.safe_load(Path(data_yaml).read_text())
    names = d["names"]
    return dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}


def subset_yaml(data_yaml, images, out_yaml) -> str:
    """Write a data.yaml whose test split is just `images` (a list of image paths)."""
    import yaml
    d = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(d.get("path", Path(data_yaml).parent))
    out_yaml = Path(out_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    lst = out_yaml.with_suffix(".txt")
    lst.write_text("\n".join(str(Path(p).resolve()) for p in images) + "\n")
    d.update(path=str(root.resolve()), test=str(lst.resolve()))
    out_yaml.write_text(yaml.safe_dump(d, sort_keys=False))
    return str(out_yaml)


def seed_of(path) -> int | None:
    m = re.search(r"_s(\d+)", str(path))
    return int(m.group(1)) if m else None


def train_imgsz(weights) -> int:
    """The image size a checkpoint was trained at (evaluate each model at its own size)."""
    import torch
    ck = torch.load(str(weights), map_location="cpu", weights_only=False)
    return int((ck.get("train_args") or {}).get("imgsz", IMGSZ))
