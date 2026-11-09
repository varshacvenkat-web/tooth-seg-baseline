"""predict.py edge cases. A fake model stands in for YOLO, so no weights are needed."""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from shapely.geometry import Polygon

import predict
from src.common import IMGSZ, letterbox, load_image

NAMES = {0: "8", 1: "9", 2: "14", 3: "13(polygon)"}


def meta_for(h, w):
    return letterbox(np.zeros((h, w, 3), np.uint8))[1]


def square(cx, cy, r, S=IMGSZ):
    m = np.zeros((S, S), np.float32)
    m[cy - r:cy + r, cx - r:cx + r] = 1
    return m


def test_empty_prediction():
    teeth, masks, labels, info = predict.postprocess(np.zeros((0, IMGSZ, IMGSZ)), [], [], NAMES, meta_for(700, 1500))
    assert teeth == [] and masks.shape == (0, 700, 1500) and labels.shape == (700, 1500) and not labels.any()


def test_masks_at_original_resolution_and_valid_polygons():
    meta = meta_for(700, 1500)
    teeth, masks, labels, _ = predict.postprocess(np.stack([square(200, 320, 40), square(400, 320, 40)]),
                                                  [0.9, 0.8], [0, 1], NAMES, meta)
    assert masks.shape == (2, 700, 1500) and masks.dtype == bool
    assert labels.shape == (700, 1500) and set(np.unique(labels)) == {0, 8, 9}
    for t, m in zip(teeth, masks):
        poly = Polygon(t["polygon"])
        assert len(t["polygon"]) >= 3 and poly.is_valid and poly.exterior.is_simple
        drawn = np.zeros((700, 1500), np.uint8)
        cv2.fillPoly(drawn, [np.round(t["polygon"]).astype(np.int32)], 1)
        assert (drawn.astype(bool) & m).sum() / (drawn.astype(bool) | m).sum() > 0.9   # same frame as the mask


def test_overlaps_go_to_the_more_confident_tooth():
    teeth, masks, labels, info = predict.postprocess(np.stack([square(300, 320, 50), square(340, 320, 50)]),
                                                     [0.6, 0.9], [0, 1], NAMES, meta_for(640, 640))
    assert info["overlap_pixels_resolved"] > 0
    assert not (masks[0] & masks[1]).any()                      # no pixel in two teeth
    assert labels[320, 320] == 9                                # the overlap belongs to the 0.9 tooth


def test_tiny_masks_are_dropped():
    teeth, masks, _, info = predict.postprocess(np.stack([square(300, 300, 60), square(100, 100, 2)]),
                                                [0.9, 0.9], [0, 1], NAMES, meta_for(640, 640), min_area_frac=0.0005)
    assert [t["tooth"] for t in teeth] == [8] and info["dropped_small"] == 1


def test_confidence_threshold_changes_counts():
    m = np.stack([square(150, 300, 30), square(300, 300, 30), square(450, 300, 30)])
    n = [len(predict.postprocess(m, [0.3, 0.6, 0.9], [0, 1, 2], NAMES, meta_for(640, 640), conf_thresh=c)[0])
         for c in (0.25, 0.5, 0.75)]
    assert n == [3, 2, 1]


def test_mask_to_polygon_handles_pinched_and_empty_masks():
    pinched = np.zeros((100, 100), bool)
    pinched[10:50, 10:50] = True
    pinched[50:90, 50:90] = True                                # touches at one corner
    poly = predict.mask_to_polygon(pinched)
    assert poly is not None and Polygon(poly).is_valid and len(poly) >= 3
    assert predict.mask_to_polygon(np.zeros((50, 50), bool)) is None


class Boxes:
    def __init__(self, conf, cls):
        self.conf, self.cls = torch.tensor(conf), torch.tensor(cls)

    def __len__(self):
        return len(self.conf)


class FakeModel:
    """Stands in for YOLO: returns one mask computed from the letterboxed image it is given."""
    names = NAMES

    def __init__(self, mask_fn):
        self.mask_fn = mask_fn

    def predict(self, img, **kw):
        m = self.mask_fn(img)
        return [SimpleNamespace(masks=SimpleNamespace(data=torch.tensor(m[None])), boxes=Boxes([0.9], [2.0]))]


def test_end_to_end_alignment_with_fake_model():
    h, w = 700, 1500
    img = np.full((h, w, 3), 40, np.uint8)
    img[200:400, 600:800] = 220                                 # a bright "tooth"
    model = FakeModel(lambda lb: (lb[..., 0] > 128).astype(np.float32))
    teeth, masks, labels, _ = predict.predict_array(model, img)
    truth = np.zeros((h, w), bool)
    truth[200:400, 600:800] = True
    assert teeth[0]["tooth"] == 14
    assert (masks[0] & truth).sum() / (masks[0] | truth).sum() > 0.95


def test_bad_inputs_give_clear_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_image(tmp_path / "missing.jpg")
    bad = tmp_path / "corrupt.jpg"
    bad.write_bytes(b"\xff\xd8 not really a jpeg")
    with pytest.raises(ValueError, match="corrupted"):
        load_image(bad)
    small = tmp_path / "small.png"
    cv2.imwrite(str(small), np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError, match="at least 32"):
        load_image(small)


def test_cli_reports_missing_weights_and_inputs(tmp_path):
    with pytest.raises(SystemExit, match="Weights not found"):
        predict.main([str(tmp_path), "--weights", str(tmp_path / "none.pt")])


def test_save_writes_all_outputs(tmp_path):
    h, w = 300, 400
    img = np.full((h, w, 3), 50, np.uint8)
    teeth, masks, labels, info = predict.postprocess(np.stack([square(320, 320, 60)]), [0.9], [0], NAMES, meta_for(h, w))
    result = {"image": str(tmp_path / "x.png"), "width": w, "height": h, "num_teeth": len(teeth), "warnings": [],
              **info, "teeth": teeth}
    predict.save(result, img, masks, labels, tmp_path)
    assert json.loads((tmp_path / "x.json").read_text())["num_teeth"] == 1
    assert cv2.imread(str(tmp_path / "x_labels.png"), cv2.IMREAD_UNCHANGED).shape == (h, w)
    assert np.load(tmp_path / "x_masks.npz")["masks"].shape == (1, h, w)
    assert (tmp_path / "x_overlay.jpg").exists()
