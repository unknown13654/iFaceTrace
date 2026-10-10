#!/usr/bin/env python3
"""
ANOD3 (anod_v3_d3) byte-exact Python inference engine.

Subclasses EspressoInference and adds reverse-engineered implementations
of 4 Apple custom 'loopkit' kernels plus NMS:

  loopkit_97  : RPN proposal generation (anchor decode + clip + permute)
  NMS layer 1 : RPN NMS (iou=0.7, score=0.1, max=200, per_class=False)
  loopkit_169 : RoI Align (PS-RoI / RoI align variant on rcnn_att)
  loopkit_204 : RCNN box decode + roll/yaw expected-angle
  NMS layer 2 : rcnn_output_nms (per_class=True, max=50)
  loopkit_285 : final top-K + class assignment

Mission ref: workers/anod3-port/missions/M-017_loopkit_reverse.md
Sibling base: verification/scripts/espresso_inference.py
"""

from __future__ import annotations

import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))

from .espresso_inference import EspressoInference, _saturate_fp16


# --------------------------------------------------------------------------- #
# RPN anchor table (M-020/M-021 [VERIFIED]).
#
# Apple bakes the loopkit_97 table into the Vision loopkit binary; it is not
# stored in the .net JSON. The table is 31,920 fp32 corner-format anchors:
#   index = i*38*30 + j*30 + k, k = ratio_idx*6 + scale_idx.
#
# --------------------------------------------------------------------------- #

FEATURE_STRIDE = 16
FEAT_H = 28
FEAT_W = 38
ANCHORS_PER_POSITION = 30
RPN_ANCHOR_COUNT = FEAT_H * FEAT_W * ANCHORS_PER_POSITION
INPUT_H = 448
INPUT_W = 608
LOOPKIT_ANCHOR_TABLE_OFFSET = 0x21f80
LOOPKIT_ANCHOR_TABLE_PATH = os.environ.get(
    'ANOD3_LOOPKIT_DUMP',
    os.path.join(_REPO_ROOT, 'artifacts', 'raw', 'loopkit_dump', 'M021_loopkit_full.bin'),
)
ANOD3_FILTER_THRESHOLDS = np.array(
    [0.92, 0.87, 0.98, 0.98, 0.96, 1.00, 0.81, 0.84],
    dtype=np.float32,
)


# --------------------------------------------------------------------------- #
# M-028 Vision-side Stage 3+4 frame transform helpers (corrected M-028 #81).
#
# [VR-028 OBSERVED] reverse-engineered from ghidra
#   - resizeAndProcessVImage           : Stage 3 un-pad + un-rotate (CCW selfie)
#   - ShotflowDetector::processBoxes   : Stage 4 terminal Y-flip + pixel scale
#
# [M-028 #81 / ghidra @ 0x1a0cfe3d8 fdiv s4,s4,s5 VERIFIED]:
#   sx, sy use the ``padded / scaled`` direction (expand, ~1.020 for 12px pad),
#   NOT ``scaled / padded`` (shrink, ~0.980). The earlier helper had the ratio
#   inverted; this revision restores the iOS-faithful Stage 3+4 transform.
# --------------------------------------------------------------------------- #


def anod3_pixel_box_to_norm_xywh(box_xyxy, *, input_w=608.0, input_h=448.0):
    """ANE-frame pixel xyxy -> normalized xywh in the padded landscape frame."""
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    return {
        "x": x1 / input_w,
        "y": y1 / input_h,
        "w": (x2 - x1) / input_w,
        "h": (y2 - y1) / input_h,
    }


def anod3_norm_box_to_shotflow_pixel_yup(
    box_xywh,
    *,
    scaled_w=596.0,
    scaled_h=448.0,
    input_w=608.0,
    input_h=448.0,
    portrait=True,
):
    """Apply Vision Stage 3 (un-pad + un-rotate) + Stage 4 (Y-flip + pixel scale).

    Input  : normalized xywh in the padded landscape frame (Y-down).
    Output : pixel xywh in the portrait frame, Y-up convention where
             ``y`` is the bottom edge of the box (matches Vision CGRect).
    """
    # [M-028 #81 / ghidra @ 0x1a0cfe3d8 VERIFIED] sx, sy = padded / scaled
    sx = input_w / scaled_w
    sy = input_h / scaled_h

    new_x = box_xywh["x"] * sx
    new_y_inter = 1.0 - ((1.0 - box_xywh["y"]) * sy)
    new_w = box_xywh["w"] * sx
    new_h = box_xywh["h"] * sy

    if portrait:
        portrait_x = new_y_inter
        portrait_y = 1.0 - (new_x + new_w)
        portrait_w = new_h
        portrait_h = new_w
        out_w = scaled_h
        out_h = scaled_w
    else:
        portrait_x = new_x
        portrait_y = new_y_inter
        portrait_w = new_w
        portrait_h = new_h
        out_w = scaled_w
        out_h = scaled_h

    return {
        "x": portrait_x * out_w,
        "y": out_h - (portrait_y + portrait_h) * out_h,
        "w": portrait_w * out_w,
        "h": portrait_h * out_h,
    }


def anod3_pixel_box_to_v6_m028(
    box_xyxy,
    *,
    scaled_w=596.0,
    scaled_h=448.0,
    input_w=608.0,
    input_h=448.0,
    portrait=True,
):
    """ANE-frame pixel xyxy -> V6 Y-up normalized xywh via Stage 3+4.

    V6 convention: x is left edge, y is BOTTOM edge in Y-up, both normalized
    over the portrait frame (W=scaled_h, H=scaled_w when ``portrait=True``).
    """
    norm = anod3_pixel_box_to_norm_xywh(box_xyxy, input_w=input_w, input_h=input_h)
    pix = anod3_norm_box_to_shotflow_pixel_yup(
        norm,
        scaled_w=scaled_w,
        scaled_h=scaled_h,
        input_w=input_w,
        input_h=input_h,
        portrait=portrait,
    )
    if portrait:
        out_w = scaled_h
        out_h = scaled_w
    else:
        out_w = scaled_w
        out_h = scaled_h
    return {
        "x": pix["x"] / out_w,
        "y": pix["y"] / out_h,
        "w": pix["w"] / out_w,
        "h": pix["h"] / out_h,
    }


def load_baked_rpn_anchors(path=LOOPKIT_ANCHOR_TABLE_PATH):
    """Load M-021 loopkit_97 anchors as (x1, y1, x2, y2) fp32."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"loopkit anchor table dump not found: {path}")
    anchors = np.fromfile(
        path,
        dtype='<f4',
        count=RPN_ANCHOR_COUNT * 4,
        offset=LOOPKIT_ANCHOR_TABLE_OFFSET,
    )
    expected = RPN_ANCHOR_COUNT * 4
    if anchors.size != expected:
        raise RuntimeError(f"anchor table too short: got {anchors.size} floats, expected {expected}")
    return anchors.reshape(RPN_ANCHOR_COUNT, 4).astype(np.float32)


def _iou_pairwise(box, others):
    """IoU of one (4,) box against (N, 4) array. Returns (N,) float32."""
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    iw = np.clip(x2 - x1, 0.0, None)
    ih = np.clip(y2 - y1, 0.0, None)
    inter = iw * ih
    a1 = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    a2 = np.clip((others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1]), 0.0, None)
    union = a1 + a2 - inter
    out = np.zeros_like(union, dtype=np.float32)
    np.divide(inter, union, out=out, where=union > 0)
    return out


def _greedy_nms(boxes, scores, iou_threshold, max_boxes):
    """Greedy NMS: returns indices into `boxes` (in selection order)."""
    order = np.argsort(-scores, kind='stable')
    keep = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(int(idx))
        if len(keep) >= max_boxes:
            break
        ious = _iou_pairwise(boxes[idx], boxes)
        suppressed |= (ious > iou_threshold)
    return np.asarray(keep, dtype=np.int64)


def _run_nms(boxes, scores, iou_threshold, score_threshold,
             num_pre_nms, max_boxes, per_class):
    """Run NMS over (boxes, scores).

    Args:
      boxes  : (N, 4) x1y1x2y2 float32
      scores : (N, S) float32 — S=1 (single-class) or S>1 (per-class scores)
      per_class:
        False — use scores[:, 0] (or max across S if S>1) as a single score.
        True  — for each class c in 0..S-1, run NMS independently and merge.

    Returns:
      keep_indices : (n_kept,) int64 — indices into `boxes`
      keep_scores  : (n_kept, S) float32 — score row for each kept index
    """
    N, S = scores.shape
    if not per_class:
        # Single-class score: take column 0 (RPN has S=1)
        s = scores[:, 0] if S == 1 else scores.max(axis=1)
        # Score threshold + pre-NMS top-K
        valid = s >= score_threshold
        if not np.any(valid):
            return np.zeros(0, dtype=np.int64), np.zeros((0, S), dtype=np.float32)
        idx_valid = np.nonzero(valid)[0]
        s_valid = s[idx_valid]
        if num_pre_nms < len(idx_valid):
            top = np.argpartition(-s_valid, num_pre_nms - 1)[:num_pre_nms]
            idx_valid = idx_valid[top]
            s_valid = s[idx_valid]
        # Sort descending
        order = np.argsort(-s_valid, kind='stable')
        idx_sorted = idx_valid[order]
        kept_local = _greedy_nms(boxes[idx_sorted], s[idx_sorted],
                                  iou_threshold, max_boxes)
        keep = idx_sorted[kept_local]
        return keep, scores[keep]

    # Per-class NMS — run NMS independently for each class column,
    # collect (idx, class, score), sort by score desc, dedupe by idx, take top max_boxes.
    all_keep = []
    for c in range(S):
        s = scores[:, c]
        valid = s >= score_threshold
        if not np.any(valid):
            continue
        idx_valid = np.nonzero(valid)[0]
        s_valid = s[idx_valid]
        if num_pre_nms and num_pre_nms < len(idx_valid):
            top = np.argpartition(-s_valid, num_pre_nms - 1)[:num_pre_nms]
            idx_valid = idx_valid[top]
            s_valid = s[idx_valid]
        order = np.argsort(-s_valid, kind='stable')
        idx_sorted = idx_valid[order]
        kept_local = _greedy_nms(boxes[idx_sorted], s[idx_sorted],
                                  iou_threshold, max_boxes)
        for k in kept_local:
            all_keep.append((int(idx_sorted[k]), c, float(s[idx_sorted[k]])))
    # Sort merged across classes by score desc; keep box with highest score on duplicate idx
    all_keep.sort(key=lambda r: -r[2])
    seen = set()
    final = []
    for idx, c, sc in all_keep:
        if idx in seen:
            continue
        seen.add(idx)
        final.append(idx)
        if len(final) >= max_boxes:
            break
    keep = np.asarray(final, dtype=np.int64)
    return keep, scores[keep] if len(keep) else np.zeros((0, S), dtype=np.float32)


class ANOD3Inference(EspressoInference):
    """Byte-exact inference of anod_v3_d3 detector.

    Most layers are inherited from EspressoInference. The subclass overrides
    `run()` only to dispatch the two custom op types this base class skips
    (`loopkit`, `non_maximum_suppression`). Everything else is the parent
    implementation verbatim, so any future improvements there propagate.
    """

    # --------------------------------------------------------------------- #
    # main loop — clone of base.run() plus loopkit + NMS dispatch
    # --------------------------------------------------------------------- #
    def run(self, input_data, output_layers=None, verbose=False, input_name=None):
        output_layers = output_layers or ['rcnn_output', 'rcnn_output_cls', 'rcnn_output_scores']
        key = input_name or 'data'
        tensors = {key: input_data.astype(np.float16)}
        outputs = {}

        total = len(self.parser.layers)
        for i, layer in enumerate(self.parser.layers):
            name = layer['name']
            lt = layer['type']
            bottom = layer.get('bottom', '')
            top = layer.get('top', name)

            if lt == 'convolution':
                self._exec_conv(tensors, layer, name, bottom, top)
            elif lt == 'inner_product':
                self._exec_fc(tensors, layer, name, bottom, top)
            elif lt == 'pool':
                self._exec_pool(tensors, layer, bottom, top)
            elif lt == 'activation':
                self._exec_activation(tensors, layer, bottom, top)
            elif lt == 'elementwise':
                self._exec_elementwise(tensors, layer, bottom, top)
            elif lt == 'softmax':
                self._exec_softmax(tensors, bottom, top)
            elif lt == 'slice':
                self._exec_slice(tensors, layer, bottom, top)
            elif lt == 'concat':
                self._exec_concat(tensors, layer, bottom, top)
            elif lt == 'reshape':
                self._exec_reshape(tensors, layer, bottom, top)
            elif lt == 'transpose':
                self._exec_transpose(tensors, layer, bottom, top)
            elif lt == 'upsample':
                self._exec_upsample(tensors, layer, bottom, top)
            elif lt == 'load_constant':
                self._exec_load_constant(tensors, layer, top)
            elif lt == 'flatten':
                self._exec_flatten(tensors, layer, bottom, top)
            elif lt == 'crop':
                self._exec_crop(tensors, layer, bottom, top)
            elif lt == 'deconvolution':
                self._exec_deconv(tensors, layer, bottom, top)
            elif lt == 'loopkit':
                self._exec_loopkit(tensors, layer, bottom, top)
            elif lt == 'non_maximum_suppression':
                self._exec_nms(tensors, layer, bottom, top)
            elif lt == 'reduce':
                # Not used in ANOD3's main detection path; placeholder.
                tensors[top] = None
            else:
                if verbose:
                    print(f"  [{i+1}/{total}] (UNHANDLED layer type {lt!r}) top={top}")
                tops = [t.strip() for t in top.split(',')] if top else []
                for t in tops:
                    tensors[t] = None

            if verbose and (i % 25 == 0 or top.split(',')[0] in output_layers):
                first_top = top.split(',')[0]
                t = tensors.get(first_top)
                if t is None:
                    print(f"  [{i+1}/{total}] {lt:20s} {first_top}: <None>")
                else:
                    arr = np.asarray(t)
                    maxv = float(np.nanmax(np.abs(arr.astype(np.float32))))
                    print(f"  [{i+1}/{total}] {lt:20s} {first_top}: shape={arr.shape} "
                          f"max_abs={maxv:.3f}")

            for t_name in (top.split(',') if top else []):
                t_name = t_name.strip()
                if t_name in output_layers:
                    val = tensors.get(t_name)
                    outputs[t_name] = None if val is None else np.asarray(val).astype(np.float32)

            if len(outputs) == len(output_layers):
                break

        return outputs

    # --------------------------------------------------------------------- #
    # loopkit dispatch
    # --------------------------------------------------------------------- #
    def _exec_loopkit(self, tensors, layer, bottom, top):
        func = layer.get('func_name', '') or ''
        tops = [t.strip() for t in top.split(',')] if top else []

        if 'nomask_97' in func:
            self._loopkit_97_rpn_proposal(tensors, layer, bottom, tops)
        elif 'nomask_169' in func:
            self._loopkit_169_roi_align(tensors, layer, bottom, tops)
        elif 'nomask_204' in func:
            self._loopkit_204_box_decode(tensors, layer, bottom, tops)
        elif 'nomask_285' in func:
            self._loopkit_285_final_topk(tensors, layer, bottom, tops)
        else:
            for t in tops:
                tensors[t] = None

    # --------------------------------------------------------------------- #
    # Override _exec_reshape to handle Espresso 'seq' dim: when a reshape
    # sets dst_seq=-1 it means preserve the per-box sequence axis. Base
    # impl treats input as 4D NCHW and fails when batch (200) is at axis 0
    # of a 2D tensor flattened from RoI features.
    # --------------------------------------------------------------------- #
    def _exec_reshape(self, tensors, layer, bottom, top):
        x = tensors.get(bottom)
        if x is None:
            tensors[top] = None
            return

        has_seq = 'dst_seq' in layer
        # Case 1: 2D RoI-feature reshape — preserve N along seq axis
        if has_seq and x.ndim == 2 and x.shape[0] > 1:
            dst_k = int(layer.get('dst_k', 0)) or x.shape[1]
            dst_h = int(layer.get('dst_h', 1)) or 1
            dst_w = int(layer.get('dst_w', 1)) or 1
            N = x.shape[0]
            tensors[top] = x.reshape(N, dst_k, dst_h, dst_w)
            return
        # Case 2: 4D RoI feature batch (200, 5, 7, 7) when reshape says
        # dst_n=-1 / dst_seq=-1 — keep batch as-is.
        if has_seq and x.ndim == 4 and x.shape[0] > 1:
            dst_k = int(layer.get('dst_k', 0)) or x.shape[1]
            dst_h = int(layer.get('dst_h', 1)) or 1
            dst_w = int(layer.get('dst_w', 1)) or 1
            N = x.shape[0]
            tensors[top] = x.reshape(N, dst_k, dst_h, dst_w)
            return

        # Fall through to base implementation
        super()._exec_reshape(tensors, layer, bottom, top)

    # --------------------------------------------------------------------- #
    # Override _exec_conv to handle batch>1 (RoI feature batches of 200).
    # Trick: merge (N, C, H, W) → (1, C, N*H, W) so the base's batch=1
    # impl handles the whole batch in one call; split back after.
    # Valid when stride doesn't cross batch boundaries — true for 1×1 convs
    # which is what bbox_fc1_1st / pose_fc1_1st use on RoI features.
    # --------------------------------------------------------------------- #
    def _exec_conv(self, tensors, layer, name, bottom, top):
        x = tensors.get(bottom)
        if x is None:
            tensors[top] = None
            return
        if x.shape[0] == 1:
            super()._exec_conv(tensors, layer, name, bottom, top)
            return

        N, C, H, W = x.shape
        # transpose to (1, C, N, H, W) → (1, C, N*H, W)
        x_merged = x.transpose(1, 0, 2, 3).reshape(1, C, N * H, W)
        tensors[bottom + '__merged'] = x_merged
        # Run base on merged
        original_bottom = bottom
        # Temporary swap so base reads merged tensor
        tensors[bottom] = x_merged
        try:
            super()._exec_conv(tensors, layer, name, bottom, top)
        finally:
            tensors[bottom] = x  # restore
            tensors.pop(bottom + '__merged', None)
        out_merged = tensors[top]  # (1, C_out, N*H_out, W_out)
        _, C_out, NH_out, W_out = out_merged.shape
        H_out = NH_out // N
        # Split back to (N, C_out, H_out, W_out)
        out = (out_merged
               .reshape(1, C_out, N, H_out, W_out)
               .transpose(0, 2, 1, 3, 4)
               .reshape(N, C_out, H_out, W_out))
        tensors[top] = out

    # --------------------------------------------------------------------- #
    # Override _exec_pool to handle batch>1 (RoI feature batches of 200)
    # Base impl in espresso_inference.py hardcodes batch=1 in the output buffer.
    # --------------------------------------------------------------------- #
    def _exec_pool(self, tensors, layer, bottom, top):
        x = tensors[bottom]
        if x is None:
            tensors[top] = None
            return
        is_global = bool(layer.get('is_global', 0))
        avg_or_max = layer.get('avg_or_max', 0)

        if is_global:
            if avg_or_max == 0:
                out = _saturate_fp16(np.mean(x.astype(np.float32), axis=(2, 3), keepdims=True))
            else:
                out = np.max(x, axis=(2, 3), keepdims=True)
            tensors[top] = out
            return

        kx = int(layer.get('size_x', 1))
        ky = int(layer.get('size_y', 1))
        sx = int(layer.get('stride_x', 1))
        sy = int(layer.get('stride_y', 1))
        N, C, H, W = x.shape
        H_out = (H - ky) // sy + 1
        W_out = (W - kx) // sx + 1
        out = np.zeros((N, C, H_out, W_out), dtype=np.float16)
        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, :, i*sy:i*sy+ky, j*sx:j*sx+kx]
                if avg_or_max == 0:
                    out[:, :, i, j] = _saturate_fp16(np.mean(patch.astype(np.float32), axis=(2, 3)))
                else:
                    out[:, :, i, j] = np.max(patch, axis=(2, 3))
        tensors[top] = out

    # --------------------------------------------------------------------- #
    # loopkit_97 — RPN proposal generation
    # --------------------------------------------------------------------- #
    _anchor_cache = None  # (cx, cy, w, h) shape (31920, 4), float32

    def _get_anchors_cwh(self):
        """Return cached anchors as (cx, cy, w, h) for fast decode."""
        if self._anchor_cache is None:
            xyxy = load_baked_rpn_anchors()  # (31920, 4) x1y1x2y2
            ax1, ay1, ax2, ay2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
            aw = ax2 - ax1
            ah = ay2 - ay1
            cx = ax1 + 0.5 * aw
            cy = ay1 + 0.5 * ah
            self._anchor_cache = np.stack([cx, cy, aw, ah], axis=-1).astype(np.float32)
        return self._anchor_cache

    def _loopkit_97_rpn_proposal(self, tensors, layer, bottom, tops):
        """RPN delta + score → image-space proposals (anchor decode + clip).

        bottom: rpn_bbox_delta (1,120,28,38), rpn_cls_score_reshape (1,60,28,38)
        tops:
          [0] clip_boxes_expand_dim              (1, 1, 31920, 4)  pixel x1y1x2y2
          [1] rpn_cls_score_reshape_expand_dim   (1, 1, 31920, 1)  fg scores flat
          [2] rpn_cls_score_reshape_neg_scores   (1, 30, 28, 38)   bg scores

        Score channel layout (block-major, derived from rpn_cls_logit
        reshape (60→2×840) + softmax along C=2):
          cls_score[0:30]  = bg probs per anchor
          cls_score[30:60] = fg probs per anchor

        Bbox delta channel layout (M-020 [VERIFIED]):
          bbox_delta[k*4 + 0] = dx for anchor k
          bbox_delta[k*4 + 1] = dy
          bbox_delta[k*4 + 2] = dw
          bbox_delta[k*4 + 3] = dh

        Anchor index ordering across the 31,920 axis: raster (h, w, anchor)
        with anchor changing fastest.
        """
        bottoms = [b.strip() for b in bottom.split(',')]
        rpn_delta = tensors.get(bottoms[0])  # (1, 120, 28, 38)
        rpn_cls = tensors.get(bottoms[1])    # (1, 60, 28, 38)

        if rpn_delta is None or rpn_cls is None:
            for t in tops:
                tensors[t] = None
            return

        delta = np.asarray(rpn_delta).astype(np.float32)
        cls = np.asarray(rpn_cls).astype(np.float32)

        # Block-major split
        bg = cls[0, 0:30]   # (30, 28, 38)
        fg = cls[0, 30:60]  # (30, 28, 38)

        # Reorder to (H, W, A) raster — anchor changes fastest
        def _flat(t):
            return t.transpose(1, 2, 0).reshape(-1)

        delta_flat = delta[0].reshape(ANCHORS_PER_POSITION, 4, FEAT_H, FEAT_W)
        delta_flat = delta_flat.transpose(2, 3, 0, 1).reshape(-1, 4)
        dx_flat = delta_flat[:, 0]
        dy_flat = delta_flat[:, 1]
        dw_flat = delta_flat[:, 2]
        dh_flat = delta_flat[:, 3]
        fg_flat = _flat(fg)

        # Anchor decode (Faster R-CNN standard, no bbox stddev normalization).
        anchors = self._get_anchors_cwh()       # (31920, 4) cx,cy,w,h
        acx, acy, aw, ah = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]

        cx = acx + dx_flat * aw
        cy = acy + dy_flat * ah
        w = aw * np.exp(dw_flat)
        h = ah * np.exp(dh_flat)

        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h

        # Clip to pixel-inclusive image bounds (608×448 ANE-native frame).
        x1 = np.clip(x1, 0.0, float(INPUT_W - 1))
        y1 = np.clip(y1, 0.0, float(INPUT_H - 1))
        x2 = np.clip(x2, 0.0, float(INPUT_W - 1))
        y2 = np.clip(y2, 0.0, float(INPUT_H - 1))

        boxes = np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32)  # (31920, 4)

        # Output tensors with shapes from shape.json
        tensors[tops[0]] = _saturate_fp16(boxes.reshape(1, 1, 31920, 4))
        tensors[tops[1]] = _saturate_fp16(fg_flat.reshape(1, 1, 31920, 1))
        tensors[tops[2]] = _saturate_fp16(bg.reshape(1, 30, 28, 38))

    def _loopkit_169_roi_align(self, tensors, layer, bottom, tops):
        """PS-RoI Align with 2×2 sub-pool (hypothesis).

        bottom: nms_boxes_pre (1, 1, 200, 4) — pixel x1y1x2y2 in 608×448
                rcnn_att     (1, 245, 28, 38) — 245 = 5 base × 7×7 PS grid

        tops:
          [0] nms_boxes  (1, 1, 200, 4)  passthrough of nms_boxes_pre
          [1] roi_align_ (200, 5, 14, 14) per box — 5 base ch, 14×14 (= 7 PS × 2 sub)

        Algorithm (PS-RoI Align, R-FCN-style with 2×2 sub-pool extension):
          For output (n, k, h_out, w_out):
            ps_i = h_out // 2,   sub_i = h_out % 2
            ps_j = w_out // 2,   sub_j = w_out % 2
            channel C = k * 49 + ps_i * 7 + ps_j     [VERIFIED]
            Sample feature C at (y_feat, x_feat) where:
              y_feat = (y1 + (h_out + 0.5) * (y2-y1) / pool_h) / 16
              x_feat = (x1 + (w_out + 0.5) * (x2-x1) / pool_w) / 16
            Bilinear interpolation in rcnn_att[C].

        Channel ordering [VERIFIED] via empirical sweep on IMG_0133:
        formula B (k*49 + ps_i*7 + ps_j) yields face_max_prob 0.939 vs
        0.005 for alternatives — k-major / R-FCN-style layout confirmed.
        """
        bottoms = [b.strip() for b in bottom.split(',')]
        boxes_t = tensors.get(bottoms[0])
        feat_t = tensors.get(bottoms[1])

        if boxes_t is None or feat_t is None:
            for t in tops:
                tensors[t] = None
            return

        boxes = np.asarray(boxes_t).astype(np.float32).reshape(-1, 4)  # (200, 4)
        feat = np.asarray(feat_t).astype(np.float32).reshape(245, 28, 38)

        N = boxes.shape[0]
        K = 5
        PS = 7
        pool_h = pool_w = 14
        H_FEAT, W_FEAT = 28, 38
        SCALE = 1.0 / 16.0  # feature stride

        # M-022 [OBSERVED] Apple variant:
        # apply half-pixel inset in image coords, then sample one point per
        # 14x14 bin from the post-inset box with a minimum 16px span.
        x1_eff = boxes[:, 0] + 0.5
        y1_eff = boxes[:, 1] + 0.5
        x2_eff = boxes[:, 2] - 0.5
        y2_eff = boxes[:, 3] - 0.5
        bw = np.maximum(x2_eff - x1_eff, 16.0)
        bh = np.maximum(y2_eff - y1_eff, 16.0)

        out = np.zeros((N, K, pool_h, pool_w), dtype=np.float32)

        # Iterate output (h, w) — for each, vectorized over 200 boxes
        for h_out in range(pool_h):
            for w_out in range(pool_w):
                ps_i = h_out // 2
                ps_j = w_out // 2

                # Feature-coord sample location (per box). sampling_ratio=1.
                y_s = (y1_eff + (h_out + 0.5) * bh / pool_h) * SCALE
                x_s = (x1_eff + (w_out + 0.5) * bw / pool_w) * SCALE

                # Bilinear corners, clamped to feature bounds
                y_lo = np.floor(y_s).astype(np.int64)
                x_lo = np.floor(x_s).astype(np.int64)
                y_hi = y_lo + 1
                x_hi = x_lo + 1
                fy = y_s - y_lo
                fx = x_s - x_lo

                # Clamp + zero-out invalid (out-of-bound boxes contribute 0)
                in_bound = (y_lo >= -1) & (y_lo <= H_FEAT) & (x_lo >= -1) & (x_lo <= W_FEAT)
                y_lo_c = np.clip(y_lo, 0, H_FEAT - 1)
                y_hi_c = np.clip(y_hi, 0, H_FEAT - 1)
                x_lo_c = np.clip(x_lo, 0, W_FEAT - 1)
                x_hi_c = np.clip(x_hi, 0, W_FEAT - 1)

                for k in range(K):
                    C = k * 49 + ps_i * 7 + ps_j   # 0..244, k-major / R-FCN [VERIFIED]
                    feat_c = feat[C]               # (28, 38)

                    v00 = feat_c[y_lo_c, x_lo_c]
                    v01 = feat_c[y_lo_c, x_hi_c]
                    v10 = feat_c[y_hi_c, x_lo_c]
                    v11 = feat_c[y_hi_c, x_hi_c]
                    v = (v00 * (1 - fy) * (1 - fx)
                         + v01 * (1 - fy) * fx
                         + v10 * fy * (1 - fx)
                         + v11 * fy * fx)
                    v = np.where(in_bound, v, 0.0)
                    out[:, k, h_out, w_out] = v

        # nms_boxes — passthrough of input boxes (typical for RoI pool layers)
        tensors[tops[0]] = boxes_t  # keep same fp16 buffer
        # roi_align_ — 4D (200, 5, 14, 14) so downstream pool_0 (size_x=2 stride=2)
        # naturally reduces to (200, 5, 7, 7)
        tensors[tops[1]] = _saturate_fp16(out)

    def _loopkit_204_box_decode(self, tensors, layer, bottom, tops):
        """Step 6: bbox-reg decode + roll/yaw expectation.

        bottom = 'bbox_reg_delta_1st,cls_prob,nms_boxes,roll_prob,yaw_prob'
        tops   = ['cls_prob_expand_dim', 'decoded_boxes_expand_dim',
                  'roll_output', 'yaw_output']
        """
        bottoms = [b.strip() for b in bottom.split(',')]
        bbox_t = tensors.get(bottoms[0])
        cls_t = tensors.get(bottoms[1])
        boxes_t = tensors.get(bottoms[2])
        roll_t = tensors.get(bottoms[3])
        yaw_t = tensors.get(bottoms[4])

        if bbox_t is None or cls_t is None or boxes_t is None or roll_t is None or yaw_t is None:
            for t in tops:
                tensors[t] = None
            return

        bbox_delta = np.asarray(bbox_t).astype(np.float32).reshape(-1, 8)
        cls_prob = np.asarray(cls_t).astype(np.float32).reshape(-1, 9)
        boxes = np.asarray(boxes_t).astype(np.float32).reshape(-1, 4)
        roll_prob = np.asarray(roll_t).astype(np.float32).reshape(-1, 12)
        yaw_prob = np.asarray(yaw_t).astype(np.float32).reshape(-1, 5)

        N = boxes.shape[0]
        if bbox_delta.shape[0] != N or cls_prob.shape[0] != N:
            raise ValueError(
                "loopkit_204 shape mismatch: "
                f"bbox={bbox_delta.shape} cls={cls_prob.shape} boxes={boxes.shape}"
            )

        # cls_prob has 9 columns: bg + 8 foreground classes. loopkit_204 drops bg
        # and exposes the foreground score table to the RCNN NMS layer.
        cls_fg = cls_prob[:, 1:9]

        # M-024 [OBSERVED]: nomask_204 always uses bbox deltas 4..7 with
        # stddev [0.1, 0.1, 0.2, 0.2]. There is no class-conditional group
        # selection in this kernel.
        dx = bbox_delta[:, 4] * 0.1
        dy = bbox_delta[:, 5] * 0.1
        dw = bbox_delta[:, 6] * 0.2
        dh = bbox_delta[:, 7] * 0.2

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w = np.maximum(x2 - x1, 0.0)
        h = np.maximum(y2 - y1, 0.0)
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h

        pred_cx = cx + dx * w
        pred_cy = cy + dy * h
        pred_w = w * np.exp(dw)
        pred_h = h * np.exp(dh)

        out_x1 = np.clip(pred_cx - 0.5 * pred_w + 0.5, 0.0, float(INPUT_W - 1))
        out_y1 = np.clip(pred_cy - 0.5 * pred_h + 0.5, 0.0, float(INPUT_H - 1))
        out_x2 = np.clip(pred_cx + 0.5 * pred_w - 0.5, 0.0, float(INPUT_W - 1))
        out_y2 = np.clip(pred_cy + 0.5 * pred_h - 0.5, 0.0, float(INPUT_H - 1))
        decoded = np.stack([out_x1, out_y1, out_x2, out_y2], axis=-1).astype(np.float32)

        roll_out = np.argmax(roll_prob, axis=1).astype(np.float32)
        yaw_out = np.argmax(yaw_prob, axis=1).astype(np.float32)

        tensors[tops[0]] = _saturate_fp16(cls_fg.reshape(1, 1, N, 8))
        tensors[tops[1]] = _saturate_fp16(decoded.reshape(1, 1, N, 4))
        tensors[tops[2]] = _saturate_fp16(roll_out.reshape(1, 1, N, 1))
        tensors[tops[3]] = _saturate_fp16(yaw_out.reshape(1, 1, N, 1))

    def _loopkit_285_final_topk(self, tensors, layer, bottom, tops):
        """Step 7: final top-50 + class index.

        bottom = 'rcnn_output_pre,rcnn_output_scores_pre'
        tops   = ['rcnn_output', 'rcnn_output_cls', 'rcnn_output_scores']
        """
        bottoms = [b.strip() for b in bottom.split(',')]
        boxes_t = tensors.get(bottoms[0])
        scores_t = tensors.get(bottoms[1])

        if boxes_t is None or scores_t is None:
            for t in tops:
                tensors[t] = None
            return

        boxes = np.asarray(boxes_t).astype(np.float32).reshape(-1, 4)
        scores = np.asarray(scores_t).astype(np.float32).reshape(boxes.shape[0], -1)

        cls_idx = np.argmax(scores, axis=1).astype(np.float32)  # 0..7 after bg-drop
        picked_scores = scores[np.arange(scores.shape[0]), cls_idx.astype(np.int64)]

        labels = (cls_idx + 1.0).astype(np.float32)  # 1..8, bg already dropped
        thresholds = ANOD3_FILTER_THRESHOLDS[cls_idx.astype(np.int64)]
        keep = picked_scores >= thresholds

        out_boxes = np.zeros_like(boxes, dtype=np.float32)
        out_labels = np.zeros((boxes.shape[0],), dtype=np.float32)
        out_scores = np.zeros((boxes.shape[0],), dtype=np.float32)
        keep_idx = np.nonzero(keep)[0]
        n_keep = len(keep_idx)
        if n_keep:
            out_boxes[:n_keep] = boxes[keep_idx]
            out_labels[:n_keep] = labels[keep_idx]
            out_scores[:n_keep] = picked_scores[keep_idx]

        tensors[tops[0]] = _saturate_fp16(out_boxes.reshape(1, 1, boxes.shape[0], 4))
        tensors[tops[1]] = _saturate_fp16(out_labels.reshape(1, 1, boxes.shape[0], 1))
        tensors[tops[2]] = _saturate_fp16(out_scores.reshape(1, 1, boxes.shape[0], 1))

    # --------------------------------------------------------------------- #
    # NMS dispatch (same algorithm, two configurations)
    # --------------------------------------------------------------------- #
    def _exec_nms(self, tensors, layer, bottom, top):
        """Standard NMS layer.

        ANOD3 has two NMS layers — both come back here. Layer params
        from .net (verified):
          nms_boxes_nms (RPN):   iou=0.7, score=0.1, max=200, num_pre=3000
          rcnn_output_nms:       per_class_suppression=True, max=50

        Bottom tensors:
          bottom[0] = boxes  shape (1, 1, N, 4) — pixel x1y1x2y2
          bottom[1] = scores shape (1, 1, N, S) — S=1 for RPN, S=8 (per-class) for RCNN

        Top tensors (4):
          [0] {prefix}_pre               (1, 1, max_boxes, 4)  selected boxes
          [1] {prefix}_scores / {prefix}_scores_pre (1, 1, max_boxes, S) selected scores
          [2] {prefix}_selected_indices  (1, 1, 1, max_boxes)  original indices in N
          [3] {prefix}_n_outputs         (1, 1, 1, 1)          # valid outputs
        """
        bottoms = [b.strip() for b in bottom.split(',')]
        tops = [t.strip() for t in top.split(',')] if top else []

        boxes_t = tensors.get(bottoms[0])
        scores_t = tensors.get(bottoms[1])
        if boxes_t is None or scores_t is None:
            for t in tops:
                tensors[t] = None
            return

        boxes = np.asarray(boxes_t).astype(np.float32).reshape(-1, 4)
        scores = np.asarray(scores_t).astype(np.float32)
        # scores shape — flatten leading dims to (N, S)
        if scores.ndim >= 2:
            scores = scores.reshape(-1, scores.shape[-1]) if scores.shape[-1] != boxes.shape[0] else scores.reshape(-1, 1)
        else:
            scores = scores.reshape(-1, 1)
        if scores.shape[0] != boxes.shape[0]:
            scores = scores.reshape(boxes.shape[0], -1)

        score_threshold = float(layer.get('score_threshold', 0.0))
        iou_threshold = float(layer.get('iou_threshold', 0.5))
        num_pre_nms = int(layer.get('num_boxes_pre_nms', 0)) or boxes.shape[0]
        max_boxes = int(layer.get('max_boxes', 100))
        per_class = bool(layer.get('per_class_suppression', False))

        keep_idx, keep_scores = _run_nms(
            boxes, scores,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            num_pre_nms=num_pre_nms,
            max_boxes=max_boxes,
            per_class=per_class,
        )
        n_kept = len(keep_idx)

        # Build outputs with zero-padding to fixed max_boxes
        S = scores.shape[1]
        out_boxes = np.zeros((max_boxes, 4), dtype=np.float32)
        out_scores = np.zeros((max_boxes, S), dtype=np.float32)
        out_indices = np.zeros((max_boxes,), dtype=np.float32)  # stored as float to match fp16 path
        if n_kept > 0:
            out_boxes[:n_kept] = boxes[keep_idx]
            out_scores[:n_kept] = keep_scores
            out_indices[:n_kept] = keep_idx.astype(np.float32)

        # Tops:
        #   [0] _pre boxes        shape (1, 1, max_boxes, 4)
        #   [1] _scores(_pre)     shape (1, 1, max_boxes, S)
        #   [2] _selected_indices shape (1, 1, 1, max_boxes)
        #   [3] _n_outputs        shape (1, 1, 1, 1)
        tensors[tops[0]] = _saturate_fp16(out_boxes.reshape(1, 1, max_boxes, 4))
        tensors[tops[1]] = _saturate_fp16(out_scores.reshape(1, 1, max_boxes, S))
        if len(tops) >= 3:
            tensors[tops[2]] = _saturate_fp16(out_indices.reshape(1, 1, 1, max_boxes))
        if len(tops) >= 4:
            tensors[tops[3]] = _saturate_fp16(np.array([[[[float(n_kept)]]]], dtype=np.float32))


# ------------------------------------------------------------------------- #
# CLI: dry-run sanity check (Step 1)
# ------------------------------------------------------------------------- #
def _step1_dryrun(verbose: bool = False) -> int:
    """Return 0 on success, non-zero on failure."""
    base = 'models/anod_v3_d3'
    net = f'{base}/model_opt.espresso.net'
    weights = f'{base}/model_opt.espresso.weights'

    print(f"[step1] Loading {net} …")
    eng = ANOD3Inference(net, weights)

    print("[step1] Running backbone+RPN dry-run with random input …")
    rng = np.random.default_rng(0)
    inp = rng.standard_normal((1, 3, 448, 608), dtype=np.float32)
    outs = eng.run(
        inp,
        output_layers=['rpn_bbox_delta', 'rpn_cls_score_reshape', 'rcnn_att'],
        input_name='data',
        verbose=verbose,
    )

    expected = {
        'rpn_bbox_delta': (1, 120, 28, 38),
        'rpn_cls_score_reshape': (1, 60, 28, 38),
        'rcnn_att': (1, 245, 28, 38),
    }
    rc = 0
    for k, exp_shape in expected.items():
        v = outs.get(k)
        if v is None:
            print(f"  [FAIL] {k}: None")
            rc = 1
            continue
        if tuple(v.shape) != exp_shape:
            print(f"  [FAIL] {k}: shape={v.shape} expected {exp_shape}")
            rc = 1
            continue
        nan = int(np.isnan(v).sum())
        inf = int(np.isinf(v).sum())
        amax = float(np.max(np.abs(v)))
        print(f"  [OK]   {k}: shape={v.shape} max_abs={amax:.3f} nan={nan} inf={inf}")
    return rc


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    sys.exit(_step1_dryrun(verbose=args.verbose))
