"""Shared offline ANOD3 face projection from node129 boxes and N x 8 scores.

The node129 row-argmax decoder is a host proposal, not verified native custom285.
Following it, this module reproduces the inspected iOS-16.7-series Vision
arithmetic for class filtering, face NMS and mean-pixel-side square enforcement.
Metadata describes the vImage input frame; it is not automatically full-asset
geometry. Equal-score order is stable input order (host choice, native unknown).
This module has no filesystem, model, image-decoder or device operations.
"""
from __future__ import annotations

import numpy as np

F = np.float32
THRESHOLDS = np.array([.92, .87, .98, .98, .96, 1., .81, .84], dtype=np.float32)
def fma32(a, b, c):
    """Correctly rounded scalar float32 FMA without a platform C runtime."""
    product = np.float64(F(a)) * np.float64(F(b))
    addend = np.float64(F(c))
    total = product + addend
    z = total - product
    error = (product - (total - z)) + (addend - z)
    rounded = F(total)
    high, low = np.nextafter(rounded, F(np.inf)), np.nextafter(rounded, F(-np.inf))
    if error > 0 and total == (np.float64(rounded) + np.float64(high)) * .5:
        return high
    if error < 0 and total == (np.float64(low) + np.float64(rounded)) * .5:
        return low
    return rounded


def raw_label(value):
    if not np.isfinite(value):
        raise ValueError('nonfinite raw class')
    return int(F(F(value) + F(1)))


def keep_threshold(score, label):
    return bool(np.isfinite(score) and 1 <= label <= 8 and
                F(score) >= THRESHOLDS[label-1])


def padded_box(xyxy):
    x1, y1, x2, y2 = map(F, xyxy)
    x, y = F(x1/F(608)), F(y1/F(448))
    w, h = F(F(x2/F(608))-x), F(F(y2/F(448))-y)
    hw, hh = F(w*F(.5)), F(h*F(.5))
    cx, cy = F(x+hw), F(y+hh)
    return [float(F(cx-hw)), float(F(F(F(1)-cy)-hh)), float(w), float(h)]


def inverse_box(box, meta):
    # Mixed f32/double order from resizeAndProcessVImage @0x1a0cfe3c4.
    x, y, w, h = box
    sw, sh = F(meta['scale_w']), F(meta['scale_h'])
    sx, sy = float(F(F(608)/sw)), float(F(F(448)/sh))
    x = x*sx
    y = 1.0-((1.0-y)*float(F(448)))/float(sh)
    w, h = w*sx, h*sy
    return [y, 1.0-(x+w), h, w] if meta['portrait'] else [x, y, w, h]


def square_face(box, width, height):
    x, y, w, h = box
    width, height = F(width), F(height)
    if min(width, height) <= 0:
        raise ValueError('positive source dimensions required')
    # fmul(height), fmadd(width), *0.5, divide; centers are cast to fp32.
    side = F(fma32(F(w), width, F(F(h)*height))*F(.5))
    nw, nh = F(side/width), F(side/height)
    cx, cy = F(x+w*.5), F(y+h*.5)
    return [float(F(cx-F(nw*F(.5)))), float(F(cy-F(nh*F(.5)))), float(nw), float(nh)]


def overlap(sender, other):
    # setBox does not refresh +8: sender area remains from initial padded box.
    ax, ay, aw, ah = sender['box']
    bx, by, bw, bh = other['box']
    right = min(F(F(bx)+F(bw)), F(ax+aw))
    bottom = min(F(F(by)+F(bh)), F(ay+ah))
    iw = max(F(0), F(right-max(F(bx), F(ax))))
    ih = max(F(0), F(bottom-max(F(by), F(ay))))
    inter = F(iw*ih)
    union = F(F(F(F(bw)*F(bh))+F(sender['cached_area']))-inter)
    return F(inter/union) if union > 0 else F(0)


def nms_faces(detections):
    # Class2 NMS/mergeHeadsBoxes cannot alter the face (label1) projection.
    candidates = sorted((d for d in detections if d['label'] == 1),
                        key=lambda d: -float(d['score']))
    kept = []
    for candidate in candidates:
        if not any(overlap(prior, candidate) > F(.3) for prior in kept):
            kept.append(candidate)
    return kept


def network_rows(outputs, meta):
    """Host node129 argmax followed by the native processVImage score gate."""
    boxes = np.asarray(outputs['rcnn_output_pre'], dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(outputs['rcnn_output_scores_pre'], dtype=np.float32).reshape(-1, 8)
    if len(boxes) != len(scores) or not (np.isfinite(boxes).all() and np.isfinite(scores).all()):
        raise ValueError('invalid post-model-NMS tensors')
    rows = []
    for index, (xyxy, scores8) in enumerate(zip(boxes, scores)):
        rawcls = F(np.argmax(scores8))
        score = float(scores8[int(rawcls)])
        if score < .5:
            continue
        if min(xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]) <= 0:
            raise ValueError('high-confidence degenerate box')
        padded = padded_box(xyxy)
        rows.append(dict(engine_row=index, raw_class=float(rawcls),
                         label=raw_label(rawcls), score=score, xyxy=xyxy.tolist(),
                         padded_box=padded, cached_area=float(F(padded[2]*padded[3])),
                         box=inverse_box(padded, meta)))
    return rows


def validate_meta(meta):
    for key in ('orig_w','orig_h','scale_w','scale_h'):
        if key not in meta or not np.isfinite(meta[key]) or meta[key] <= 0:
            raise ValueError('positive finite '+key+' required')
    if not isinstance(meta.get('portrait'), (bool, np.bool_)) or meta['scale_w'] > 608 or meta['scale_h'] > 448:
        raise ValueError('invalid paired resize metadata')
    if bool(meta['portrait']) != bool(F(meta['orig_h']) > F(meta['orig_w'])):
        raise ValueError('portrait flag disagrees with original float32 dimensions')


def postprocess_faces(boxes, class_scores, meta):
    """Return all label1 faces plus raw node129 and intermediate provenance.

    One image per call. Optional leading axes must be singleton.
    boxes: (N, 4) padded 608x448 pixel xyxy, not labels or V6 rectangles.
    class_scores: (N, 8), background already removed, NOT a class vector.
    meta: orig_w, orig_h, scale_w, scale_h, portrait; all dimensions positive.
    faces[*].box: squared normalized lower-left xywh in the input vImage frame.
    """
    boxes, class_scores = np.asarray(boxes), np.asarray(class_scores)
    if boxes.ndim < 2 or boxes.shape[-1] != 4 or class_scores.ndim < 2 or class_scores.shape[-1] != 8:
        raise ValueError('expected node129 N x 4 boxes and N x 8 class scores')
    if any(n != 1 for n in boxes.shape[:-2]+class_scores.shape[:-2]):
        raise ValueError('one image per call; leading axes must be singleton')
    validate_meta(meta)
    raw = network_rows({'rcnn_output_pre':boxes, 'rcnn_output_scores_pre':class_scores}, meta)
    filtered = [d for d in raw if keep_threshold(d['score'], d['label'])]
    kept = nms_faces(filtered)
    faces = [dict(d, box_before_square=d['box'],
                  box=square_face(d['box'], meta['orig_w'], meta['orig_h'])) for d in kept]
    return dict(faces=faces, raw_rows=raw, after_class_filter=filtered,
                node129_boxes=boxes.reshape(-1,4).tolist(),
                node129_class_scores=class_scores.reshape(-1,8).tolist(),
                tie_policy='stable input order; native ties unspecified')
