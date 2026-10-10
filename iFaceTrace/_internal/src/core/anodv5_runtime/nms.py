"""ANODv5 v3 host NMS: the graph NMS of the model and the Vision wrapper records and face IOM.

The definitions below are verbatim copies from the research port (anodv5_common.py,
anodv5_runtime.py, anodv5_runtime_v3.py; copies in tests/fixtures/anodv5). Only the import
lines differ. The stable descending sort in graph_nms is the tie policy that the detector
profile fixes for ANODv5; the vDSP order of the ANODv3 detector does not apply here.
"""
from __future__ import annotations

import numpy as np

from src.core.anodv5_runtime.geometry import corner_pixels_to_vision_box, require

PRE_NMS_OUTPUTS = ["detection_expand_dim", "per_box_prob_expand_dim", "fcos/human_pose",
                   "fcos/pet_face_scores", "fcos/association_preds"]
TIE_POLICY = "HOST_STABLE_DESCENDING_SCORE_THEN_INPUT_INDEX_FIRST_ARGMAX"


def post_nms_tail(boxes_center, scores, indices, count):
    """Fixed FP32 graph tail; the argmax host policy stays explicit."""
    require(boxes_center.shape == (1, 1, 50, 4), "wrong NMS box shape")
    require(scores.shape == (1, 1, 50, 12), "wrong NMS score shape")
    require(indices.shape == (1, 1, 1, 50), "wrong selected-index shape")
    require(boxes_center.dtype == scores.dtype == np.float32, "wrong NMS float dtype")
    require(isinstance(count, (int, np.integer)) and 0 <= count <= 50, "wrong NMS count")
    require(np.isfinite(boxes_center).all() and np.isfinite(scores).all(), "non-finite NMS output")
    x, y, w, h = (boxes_center[..., i:i + 1] for i in range(4))
    boxes = np.concatenate([x - w * np.float32(.5) + np.float32(.5),
                            y - h * np.float32(.5) + np.float32(.5),
                            x + w * np.float32(.5) - np.float32(.5),
                            y + h * np.float32(.5) - np.float32(.5)], axis=3)
    return dict(rcnn_output=boxes, rcnn_output_scores=np.max(scores, axis=3, keepdims=True),
                rcnn_output_cls=np.argmax(scores, axis=3, keepdims=True).astype(np.float32),
                rcnn_output_selected_indices=indices,
                rcnn_output_n_outputs=np.array([[[[count]]]], dtype=np.float32),
                interpretation="HOST_MODE9_MAX_MODE11_FIRST_ARGMAX_PENDING_NATIVE_PROOF")


def center_to_overlap(boxes):
    """Espresso26 center coordinates use nonnegative extent and no ±0.5 offset."""
    extents = np.maximum(boxes[:, 2:4], np.float32(0))
    half = extents * np.float32(.5)
    return np.concatenate([boxes[:, :2] - half, boxes[:, :2] + half], axis=1), extents


def graph_nms(boxes, scores, *, iou_threshold, score_threshold, union_epsilon,
              num_boxes_pre_nms, max_boxes, per_class_suppression, tie_policy):
    """Global row-max top-K, same-argmax-class suppression, full score-row copy.

    The tie/argmax rule is an explicit HOST policy. Numeric overlap uses FP32
    bounds/difference, FP64 intersection product, rounded FP32 areas/union,
    and FP64 division with the evidence-supplied epsilon. It is not a native
    accumulation or library-sort equality assertion.
    """
    require(boxes.ndim == scores.ndim == 2 and boxes.shape[1] == 4
            and scores.shape[0] == boxes.shape[0] and scores.shape[1] == 12,
            "wrong ANODv5 graph NMS shapes")
    require(boxes.dtype == scores.dtype == np.float32, "wrong NMS input dtype")
    require(np.isfinite(boxes).all() and np.isfinite(scores).all(), "non-finite NMS input")
    require(np.isfinite([iou_threshold, score_threshold, union_epsilon]).all()
            and 0 <= iou_threshold <= 1 and union_epsilon > 0, "invalid NMS numeric contract")
    require(num_boxes_pre_nms == 300 and max_boxes == 50 and per_class_suppression is True,
            "wrong ANODv5 NMS declaration")
    require(tie_policy == TIE_POLICY, "unsupported or implicit NMS tie policy")
    row_max = np.max(scores, axis=1)
    row_class = np.argmax(scores, axis=1)
    valid = np.flatnonzero(row_max > np.float32(score_threshold))
    order = valid[np.argsort(-row_max[valid], kind="stable")][:num_boxes_pre_nms]
    overlap_boxes, extents = center_to_overlap(boxes)
    suppressed = np.zeros(order.size, dtype=bool)
    keep = []
    borderline = []
    for position, index in enumerate(order):
        if suppressed[position]:
            continue
        keep.append(int(index))
        if len(keep) == max_boxes:
            break
        future_positions = np.arange(position + 1, len(order))
        future = order[future_positions]
        same = row_class[future] == row_class[index]
        future_positions, future = future_positions[same], future[same]
        if not len(future):
            continue
        upper = np.minimum(overlap_boxes[index, 2:], overlap_boxes[future, 2:])
        lower = np.maximum(overlap_boxes[index, :2], overlap_boxes[future, :2])
        delta = np.maximum((upper - lower).astype(np.float64), 0)
        intersection = (delta[:, 0] * delta[:, 1]).astype(np.float32)
        area_future = (extents[future, 0] * extents[future, 1]).astype(np.float32)
        area_sum = (float(extents[index, 0]) * float(extents[index, 1])
                    + area_future.astype(np.float64)).astype(np.float32)
        union = (area_sum - intersection).astype(np.float32)
        ratios = (intersection.astype(np.float64) / (union.astype(np.float64)
                  + float(union_epsilon))).astype(np.float32)
        suppress = ratios > np.float32(iou_threshold)
        suppressed[future_positions] |= suppress
        for candidate, ratio in zip(future[np.abs(ratios - iou_threshold) <= 1e-6],
                                    ratios[np.abs(ratios - iou_threshold) <= 1e-6]):
            borderline.append(dict(selected=int(index), candidate=int(candidate), iou=float(ratio)))
    kept = np.asarray(keep, dtype=np.int64)
    out_boxes = np.zeros((1, 1, 50, 4), dtype=np.float32)
    out_scores = np.zeros((1, 1, 50, 12), dtype=np.float32)
    out_indices = np.full((1, 1, 1, 50), -1, dtype=np.float32)
    out_boxes[0, 0, :len(keep)] = boxes[kept]
    out_scores[0, 0, :len(keep)] = scores[kept]
    out_indices[0, 0, 0, :len(keep)] = kept
    result = post_nms_tail(out_boxes, out_scores, out_indices, len(keep))
    result["rcnn_output_pre"] = out_boxes
    result["rcnn_output_scores_pre"] = out_scores
    result["diagnostics"] = dict(tie_policy=tie_policy, native_tie_order="UNKNOWN",
        input_count=len(boxes), score_pass_count=len(valid), sorted_pre_nms_indices=order.tolist(),
        selected_indices=keep, output_count=len(keep),
        equal_row_score_groups=int(np.sum(np.diff(row_max[order]) == 0)),
        rows_with_equal_class_max=int(np.sum(np.sum(scores[order] == row_max[order, None], axis=1) > 1)),
        near_iou_threshold=borderline, native_arithmetic_parity="NOT_MEASURED")
    return result


def vision_records_v3(tail, confidence_floor):
    """Native processVImage scans its fixed 50 rows and gates sqrt confidence."""
    boxes = tail["rcnn_output"].reshape(50, 4)
    scores = tail["rcnn_output_scores"].reshape(50)
    classes = tail["rcnn_output_cls"].reshape(50)
    indices = tail["rcnn_output_selected_indices"].reshape(50)
    require(np.float32(confidence_floor) == np.float32(.3), "wrong confidence floor")
    records, rejected = [], []
    for i in range(50):
        require(np.isfinite(scores[i]) and scores[i] >= 0, "invalid confidence input")
        confidence = np.sqrt(np.float32(scores[i]), dtype=np.float32)
        if confidence < np.float32(confidence_floor):
            rejected.append(dict(row=i, reason="confidence_floor", confidence=float(confidence)))
            continue
        require(np.isfinite(classes[i]) and classes[i] == int(classes[i])
                and 0 <= classes[i] < 12, "invalid class index")
        require(indices[i] == int(indices[i]) and 0 <= indices[i] < 3648, "invalid selected index")
        records.append(dict(row=i, class_index=int(classes[i]), label=int(np.float32(classes[i] + 1)),
            score=float(confidence), raw_score=float(scores[i]), selected_index=int(indices[i]),
            model_input_normalized_box=corner_pixels_to_vision_box(boxes[i])))
    return records, rejected


def vision_iom_v3(selected, other):
    """1a798d220 mixed arithmetic, including the asymmetric operand rounding."""
    ax, ay, aw, ah = map(float, selected)
    bx, by, bw, bh = map(float, other)
    require(np.isfinite([ax, ay, aw, ah, bx, by, bw, bh]).all()
            and min(aw, ah, bw, bh) > 0, "finite positive IOM extents required")
    other_right = np.float32(np.float32(bx) + np.float32(bw))
    other_top = np.float32(np.float32(by) + np.float32(bh))
    selected_right, selected_top = np.float32(ax + aw), np.float32(ay + ah)
    intersection_w = np.maximum(np.float32(np.minimum(other_right, selected_right)
                                  - np.maximum(np.float32(ax), np.float32(bx))), np.float32(0))
    intersection_h = np.maximum(np.float32(np.minimum(other_top, selected_top)
                                  - np.maximum(np.float32(ay), np.float32(by))), np.float32(0))
    intersection = np.float32(intersection_w * intersection_h)
    other_area = np.float32(np.float32(bw) * np.float32(bh))
    selected_area_f64 = aw * ah
    denominator = other_area if selected_area_f64 > float(other_area) else np.float32(selected_area_f64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.float32(intersection / denominator)


def vision_face_nms_v3(records, threshold):
    require(np.float32(threshold) == np.float32(.3), "wrong Vision face IOM threshold")
    faces = [r for r in records if r["class_index"] == 0]
    order = sorted(range(len(faces)), key=lambda i: (-faces[i]["score"], i))
    retained, suppressions = [], []
    for index in order:
        row = faces[index]
        for kept in retained:
            ratio = vision_iom_v3(kept["detector_crop_normalized_box"], row["detector_crop_normalized_box"])
            if ratio > np.float32(threshold):
                suppressions.append(dict(row=row["row"], retained_row=kept["row"], iom=float(ratio) if np.isfinite(ratio) else "NON_FINITE_POSITIVE_INFINITY"))
                break
        else:
            retained.append(row)
    kept_rows = {r["row"] for r in retained}
    others = [r for r in records if r["class_index"] != 0]
    return retained + others, dict(criterion="native_mixed_precision_intersection_over_min_area",
                                   suppressions=suppressions, native_tie_order="UNKNOWN", tie_policy=TIE_POLICY)
