"""ANODv5 v3 host geometry: first crop, frame fit, box mapping and the unit-rect gate.

The definitions below are verbatim copies from the research port (anodv5_common.py,
anodv5_geometry.py, anodv5_geometry_v2.py; copies in tests/fixtures/anodv5). Only the
import lines differ. tests/test_anodv5_runtime.py compares the AST with the research files.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

RESIZE_POLICY = "HOST_PIL_BILINEAR_NO_PIXEL_CALIBRATION"
ROTATION_POLICY = "HOST_NUMPY_ROT90_CCW_1_FROM_ROTATION_ARGUMENT_1"


class ContractError(ValueError):
    pass


def require(test, message):
    if not test:
        raise ContractError(message)


def frame_size(width, height):
    require(width > 0 and height > 0 and width >= height, "landscape dimensions required")
    w, h = np.float32(width), np.float32(height)
    ratio = np.float32(w / h)
    if ratio < np.float32(np.float32(608) / np.float32(384)):
        sw, sh = int(np.float32(ratio * np.float32(384))), 384
    else:
        sw, sh = 608, int(np.float32(np.float32(h / w) * np.float32(608)))
    require(1 <= sw <= 608 and 1 <= sh <= 384, "invalid resized dimensions")
    return sw, sh


def unpad_and_unrotate(box, metadata):
    """Vision26 resizeAndProcessVImage double-coordinate map, no clipping."""
    x, y, w, h = map(float, box)
    sx = metadata["input_width"] / metadata["scaled_width"]
    sy = metadata["input_height"] / metadata["scaled_height"]
    x, y, w, h = x * sx, 1 - (1 - y) * sy, w * sx, h * sy
    result = [y, 1 - (x + w), h, w] if metadata["portrait"] else [x, y, w, h]
    require(np.isfinite(result).all(), "non-finite transformed box")
    return result


def enforce_square_face(box, *, original_width, original_height):
    """Vision26 label1 arithmetic: pixel side is mean of width and height."""
    require(original_width > 0 and original_height > 0, "invalid original dimensions")
    x, y, w, h = map(float, box)
    cx, cy = np.float32(x + w * .5), np.float32(y + h * .5)
    width, height = np.float32(original_width), np.float32(original_height)
    hp = np.float32(height * np.float32(h))
    side = np.float32(np.float32(float(np.float32(w)) * float(width) + float(hp)) * np.float32(.5))
    sw, sh = np.float32(side / width), np.float32(side / height)
    xx = np.float32(cx - np.float32(sw * np.float32(.5)))
    yy = np.float32(cy - np.float32(sh * np.float32(.5)))
    require(np.isfinite([xx, yy, sw, sh]).all() and sw > 0 and sh > 0, "invalid square face")
    return list(map(float, [xx, yy, sw, sh]))


def suggested_first_crop_size(width, height):
    """Vision getSuggestedImageSize: FP64 aspect/factor, FRINTA, shrink both only."""
    require(width > 0 and height > 0, "invalid source ROI dimensions")
    short, long = min(width, height), max(width, height)
    ratio = float(long) / float(short)
    threshold = float(np.float32(np.float32(608) / np.float32(384)))
    scale = 384. / short if ratio < threshold else 608. / long
    target_w, target_h = math.floor(width * scale + .5), math.floor(height * scale + .5)
    require(target_w > 0 and target_h > 0, "invalid first-crop dimensions")
    if target_w < width and target_h < height:
        return target_w, target_h
    return width, height


def preprocess_rgb_v2(rgb, *, scale, mean, channel_order, resize_policy, rotation_policy):
    require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
            and min(rgb.shape[:2]) > 0, "nonempty UINT8 RGB required")
    require(channel_order == "BGR" and resize_policy == RESIZE_POLICY
            and rotation_policy == ROTATION_POLICY, "wrong v2 input policy")
    means = np.asarray(mean, dtype=np.float32)
    require(means.shape == (3,) and np.isfinite(means).all() and np.isfinite(scale)
            and scale > 0, "invalid affine contract")
    orig_h, orig_w = rgb.shape[:2]
    crop_w, crop_h = suggested_first_crop_size(orig_w, orig_h)
    first_crop = np.asarray(Image.fromarray(rgb).resize((crop_w, crop_h), Image.Resampling.BILINEAR))
    portrait = crop_h > crop_w
    landscape = np.rot90(first_crop, 1) if portrait else first_crop
    sw, sh = frame_size(landscape.shape[1], landscape.shape[0])
    canvas = np.zeros((384, 608, 3), dtype=np.uint8)
    canvas[:sh, :sw] = np.asarray(Image.fromarray(landscape).resize((sw, sh), Image.Resampling.BILINEAR))
    bias = (means * np.float32(scale)).astype(np.float32)
    channels = canvas[:, :, ::-1].astype(np.float32)
    tensor = np.ascontiguousarray((channels * np.float32(scale) + bias).transpose(2, 0, 1)[None])
    metadata = dict(orig_width=orig_w, orig_height=orig_h, crop_width=crop_w, crop_height=crop_h,
                    roi_pixel_xywh=[0., 0., float(orig_w), float(orig_h)], full_roi_only=True,
                    portrait=portrait, scaled_width=sw, scaled_height=sh, input_width=608, input_height=384,
                    channel_order=channel_order, scale=float(np.float32(scale)), mean=means.tolist(),
                    bias=bias.tolist(), resize_policy=resize_policy, rotation_policy=rotation_policy,
                    padding="BLACK_TOP_LEFT_ROW_COPY", pipeline_geometry_version=2,
                    first_crop="FULL_RGB_ROI_FRINTA_SHRINK_BOTH_IF_SMALLER_HOST_PIL_BILINEAR",
                    net_file_transform="RGB_SCALE1_BIAS0_BYPASSED_BY_EXPLICIT_WRAPPER_PARAMS",
                    native_crop_and_resampling_pixel_parity="NOT_MEASURED")
    return tensor, metadata, canvas


def corner_pixels_to_vision_box(box, width=608, height=384):
    """Four FP32 divisions, promoted to FP64, followed by native-order FP64 math."""
    pixels = np.asarray(box, dtype=np.float32)
    require(pixels.shape == (4,) and np.isfinite(pixels).all(), "invalid corner box")
    nx1 = float(np.float32(pixels[0] / np.float32(width)))
    ny1 = float(np.float32(pixels[1] / np.float32(height)))
    nx2 = float(np.float32(pixels[2] / np.float32(width)))
    ny2 = float(np.float32(pixels[3] / np.float32(height)))
    w, h = nx2 - nx1, ny2 - ny1
    half_w, half_h = w * .5, h * .5
    cx, cy = nx1 + half_w, ny1 + half_h
    return [cx - half_w, (1. - cy) - half_h, w, h]


def square_in_detector_crop(box, metadata):
    """Square with the first VN crop dimensions, not original full RGB dimensions."""
    return enforce_square_face(box, original_width=metadata["crop_width"],
                               original_height=metadata["crop_height"])


def detector_box_to_source_roi(box, metadata):
    """Base Detection pixel bridge then source ROI FP32 scale/FP64 geometry."""
    require(metadata["full_roi_only"] is True, "non-full ROI contract not implemented")
    x, y, w, h = map(float, box)
    cw, ch = float(metadata["crop_width"]), float(metadata["crop_height"])
    source_w, source_h = float(metadata["orig_width"]), float(metadata["orig_height"])
    roi_x, roi_y, roi_w, roi_h = metadata["roi_pixel_xywh"]
    require(roi_x == roi_y == 0 and roi_w == source_w and roi_h == source_h,
            "full source ROI required; no guessed FMA for ROI offsets")
    px, py, pw, ph = x * cw, ch - (y * ch + h * ch), w * cw, h * ch
    sx = float(np.float32(roi_w / float(np.float32(cw))))
    sy = float(np.float32(roi_h / float(np.float32(ch))))
    result = [(px * sx) / source_w, ((ch - (py + ph)) * sy) / source_h,
              (pw * sx) / source_w, (ph * sy) / source_h]
    require(np.isfinite(result).all(), "non-finite source ROI mapping")
    return result, dict(detector_pixel_top_left_xywh=[px, py, pw, ph],
                        roi_scale_f32=[sx, sy], source_normalized_lower_left_xywh=result)


def source_unit_rect_gate(box, *, semantics):
    """No clipping. Select a semantics only after its target has static proof."""
    require(semantics in ("NATIVE_CGRECT_INTERSECTS_RECT_STRICT", "NATIVE_CGRECT_CONTAINS_RECT"),
            "UNIT_RECT_GATE_NOT_ESTABLISHED")
    x, y, w, h = map(float, box)
    if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0:
        return False
    if semantics == "NATIVE_CGRECT_INTERSECTS_RECT_STRICT":
        return max(x, 0.) < min(x + w, 1.) and max(y, 0.) < min(y + h, 1.)
    return x >= 0. and y >= 0. and x + w <= 1. and y + h <= 1.
