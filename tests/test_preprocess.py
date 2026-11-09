"""Preprocessing and un-padding: shapes, dtypes and value ranges, plus parity with the
preprocessing Ultralytics applies during evaluation. If predict.py's input pipeline and the
evaluation pipeline ever diverge, test_parity_with_ultralytics_eval_loader fails."""
import cv2
import numpy as np
import pytest

from src.common import IMGSZ, PAD_VALUE, TRAIN_SETTINGS, letterbox, load_image, to_model_format, unletterbox_mask

SIZES = [(1316, 2940), (700, 1500), (900, 500), (300, 500), (640, 640)]


def xray(h, w, seed=0):
    rng = np.random.default_rng(seed)
    g = (rng.random((h, w)) * 255).astype(np.uint8)
    g = cv2.GaussianBlur(g, (0, 0), 3)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize("h,w", SIZES)
def test_letterbox_shape_dtype_range(h, w):
    img = xray(h, w)
    out, meta = letterbox(img)
    assert out.shape == (IMGSZ, IMGSZ, 3)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255
    rh, rw = meta["resized_hw"]
    assert max(rh, rw) == IMGSZ                               # long side fills the input
    assert abs(rw / rh - w / h) < 0.02                         # aspect ratio kept
    top, left = meta["pad_tl"]
    assert (out[:top] == PAD_VALUE).all() and (out[top + rh:] == PAD_VALUE).all()
    assert (out[:, :left] == PAD_VALUE).all() and (out[:, left + rw:] == PAD_VALUE).all()


@pytest.mark.parametrize("h,w", SIZES)
def test_unletterbox_round_trip(h, w):
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (w // 5, h // 4), 15, 0, 360, 1, -1)
    lb, meta = letterbox(cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR))
    back = unletterbox_mask(lb[..., 0] > 127, meta)
    assert back.shape == (h, w) and back.dtype == bool
    inter, union = (back & (mask > 0)).sum(), (back | (mask > 0)).sum()
    assert inter / union > 0.97


def test_pad_region_maps_to_nothing():
    lb, meta = letterbox(xray(700, 1500))
    only_padding = np.zeros((IMGSZ, IMGSZ), bool)
    top, _ = meta["pad_tl"]
    only_padding[:top] = True
    assert not unletterbox_mask(only_padding, meta).any()


def test_load_image_matches_ultralytics_reader(tmp_path):
    from ultralytics.utils.patches import imread
    for k, (h, w) in enumerate(SIZES[:3]):
        p = tmp_path / f"x{k}.jpg"
        cv2.imwrite(str(p), xray(h, w, k)[..., 0])       # grayscale JPEG, like the dataset
        assert np.array_equal(load_image(p), imread(str(p)))


def test_input_formats(tmp_path):
    g = xray(300, 400)[..., 0]
    cv2.imwrite(str(tmp_path / "gray.png"), g)
    cv2.imwrite(str(tmp_path / "rgba.png"), cv2.cvtColor(g, cv2.COLOR_GRAY2BGRA))
    cv2.imwrite(str(tmp_path / "16bit.png"), g.astype(np.uint16) * 257)
    colour = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    colour[..., 2] = 255 - colour[..., 2]
    cv2.imwrite(str(tmp_path / "colour.png"), colour)
    for name in ("gray.png", "rgba.png", "16bit.png", "colour.png"):
        im = load_image(tmp_path / name)
        assert im.shape == (300, 400, 3) and im.dtype == np.uint8, name
        assert np.array_equal(im[..., 0], im[..., 1]) and np.array_equal(im[..., 1], im[..., 2]), name
    assert np.array_equal(load_image(tmp_path / "16bit.png")[..., 0], g)


def test_settings_agree():
    assert TRAIN_SETTINGS["imgsz"] == IMGSZ
    import predict
    assert predict.IMGSZ == IMGSZ and predict.letterbox is letterbox


def test_parity_with_ultralytics_eval_loader(tmp_path):
    """Our letterbox content must equal what Ultralytics' validation dataset feeds the model."""
    from ultralytics.data import YOLODataset
    from ultralytics.data.augment import LetterBox
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    for k, (h, w) in enumerate(SIZES):
        cv2.imwrite(str(tmp_path / "images" / f"x{k}.png"), xray(h, w, k))
        (tmp_path / "labels" / f"x{k}.txt").write_text("0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n")
    ds = YOLODataset(img_path=str(tmp_path / "images"), imgsz=IMGSZ, augment=False, rect=True, batch_size=1,
                     data={"names": {0: "tooth"}, "channels": 3}, task="segment")
    for i, f in enumerate(ds.im_files):
        theirs, _, _ = ds.load_image(i)
        ours, meta = letterbox(load_image(f))
        (top, left), (h, w) = meta["pad_tl"], meta["resized_hw"]
        assert theirs.shape[:2] == (h, w), f"resized size differs for {f}"
        assert np.array_equal(ours[top:top + h, left:left + w], theirs), f"pixels differ for {f}"
    assert LetterBox().padding_value == PAD_VALUE


def test_to_model_format_rejects_bad_shapes():
    with pytest.raises(ValueError):
        to_model_format(np.zeros((2, 64, 64, 3), np.uint8))
    with pytest.raises(ValueError):
        to_model_format(np.zeros((64, 64, 5), np.uint8))
    with pytest.raises(ValueError):
        to_model_format(np.zeros((10, 10), np.uint8))
