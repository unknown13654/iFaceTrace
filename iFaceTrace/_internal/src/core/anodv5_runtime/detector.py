"""ANODv5 v3 host detector: one pinned pre-NMS ONNX model plus the host NMS and Vision wrapper.

This is a host port built from static analysis. It is not a claim that the whole Apple
pipeline is identical or that native bit parity was reached. validate_profile_v3 and
Anodv5DetectorV3.detect_rgb are verbatim copies from the research port (copies in
tests/fixtures/anodv5); the research gate, call budget and pre-NMS capture are not here.
"""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.config import load_settings
from src.core.anodv5_runtime.geometry import (detector_box_to_source_roi, preprocess_rgb_v2,
                                              require, source_unit_rect_gate,
                                              square_in_detector_crop, unpad_and_unrotate)
from src.core.anodv5_runtime.nms import (PRE_NMS_OUTPUTS, TIE_POLICY, graph_nms,
                                         vision_face_nms_v3, vision_records_v3)

DETECTOR_ID = "anodv5-v3"
_ONNX_NAME = "anodv5_pre_nms.onnx"
_PROFILE_NAME = "DETECTOR_PROFILE_V3.json"
_MANIFEST_NAME = "MANIFEST.json"
_RUNTIME = Path(__file__).resolve().parent
ONNX_SHA256 = "fab2320ad72efefdbbba2cd2d7fddf09dc10fe4a755abdfa590026d7bca517f9"
OUTPUT_SHAPES = {
    "detection_expand_dim": (1, 1, 3648, 4),
    "per_box_prob_expand_dim": (1, 1, 3648, 12),
    "fcos/human_pose": (1, 48, 76, 3),
    "fcos/pet_face_scores": (1, 48, 76, 1),
    "fcos/association_preds": (1, 1, 3648, 2),
}
# The pins live in code, not only in the model folder: a folder whose files agree with
# its own MANIFEST.json but are foreign must still be refused.
PINNED_ONNX_SHA256 = ONNX_SHA256
PINNED_PROFILE_SHA256 = "c1ff0bf54f8e6e2df580b4992581f0c3986d9882b41dfe48646ea1bd184a970a"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_profile_v3(profile):
    require(profile["profile_id"] == "ANODv5_PORTABLE_CONDITIONAL_CPU_BGR_FP32_HOST_BILINEAR_V3",
            "wrong v3 profile")
    p, n, wrapper = profile["preprocessing"], profile["graph_nms"], profile["wrapper"]
    require(p["channel_order"] == "BGR" and p["input_size"] == [608, 384]
            and np.float32(p["scale"]) == np.float32(.01742919348180294)
            and np.array_equal(np.float32(p["mean"]), np.float32([-103.53, -116.28, -123.675]))
            and p["first_crop_scope"] == "FULL_RGB_ROI", "wrong v3 input contract")
    require(n == dict(iou_threshold=.5, score_threshold=0., union_epsilon=1e-7,
                     num_boxes_pre_nms=300, max_boxes=50, per_class_suppression=True, tie_policy=TIE_POLICY),
            "wrong v3 graph NMS contract")
    require(np.array_equal(np.float32(wrapper["class_thresholds"]),
                           np.float32([.568, .492, .598, .559, .571, 1, .565, .532, .6, .632, .615, .543]))
            and np.asarray(wrapper["class_thresholds"], dtype="<f4").tobytes().hex()
            == "".join(wrapper["class_threshold_fp32_le_hex"]), "wrong class threshold bits")
    require(np.float32(wrapper["confidence_floor"]) == np.float32(.3)
            and np.float32(wrapper["intersection_over_min_area_threshold"]) == np.float32(.3)
            and wrapper["unit_rect_gate"] == "NATIVE_CGRECT_INTERSECTS_RECT_STRICT"
            and wrapper["square_dimensions"] == "FIRST_VN_CROP_BUFFER", "wrong v3 wrapper geometry")


class PreNMSRuntimeV3:
    """The ONNX session of the research port, without its gate, call budget and dispatch lock."""

    def __init__(self, model_path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        require([(i.name, i.shape, i.type) for i in self.session.get_inputs()]
                == [("image", [1, 3, 384, 608], "tensor(float)")], "wrong v3 input shape")
        require([o.name for o in self.session.get_outputs()] == PRE_NMS_OUTPUTS, "wrong v3 output names")
        self.runtime_version = ort.__version__

    def infer_tensor(self, tensor):
        require(tensor.shape == (1, 3, 384, 608) and tensor.dtype == np.float32
                and np.isfinite(tensor).all(), "wrong v3 FLOAT32 model input")
        values = self.session.run(PRE_NMS_OUTPUTS, {"image": np.ascontiguousarray(tensor)})
        require(len(values) == len(PRE_NMS_OUTPUTS), "wrong v3 output count")
        raw = dict(zip(PRE_NMS_OUTPUTS, values))
        for name, value in raw.items():
            require(value.shape == OUTPUT_SHAPES[name] and value.dtype == np.float32
                    and np.isfinite(value).all(), f"wrong v3 output contract: {name}")
        return raw

    def as_dict(self):
        return dict(runtime="ORT_CPU_HOST_FLOAT32_V3", runtime_version=self.runtime_version,
                    onnx_sha256=ONNX_SHA256, native_execution=False,
                    native_arithmetic_parity="NOT_MEASURED", native_sort_ties="UNKNOWN")


class Anodv5DetectorV3:
    """One pinned detector. A model, profile or runtime file that does not match is a hard failure."""

    def __init__(self, model_dir: Path | None = None) -> None:
        model_dir = Path(load_settings().anodv5_dir if model_dir is None else model_dir)
        manifest_path = model_dir / _MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, pinned in ((_ONNX_NAME, PINNED_ONNX_SHA256), (_PROFILE_NAME, PINNED_PROFILE_SHA256)):
            if (manifest.get("files", {}).get(name, {}).get("sha256") != pinned
                    or _sha256(model_dir / name) != pinned):
                raise ValueError(f"ANODv5 model file mismatch: {name}")
        if manifest.get("runtime_sha256") != {path.name: _sha256(path) for path in _RUNTIME.glob("*.py")}:
            raise ValueError("ANODv5 runtime module mismatch")
        self.profile = json.loads((model_dir / _PROFILE_NAME).read_text(encoding="utf-8"))
        validate_profile_v3(self.profile)
        self.profile_id = self.profile["profile_id"]
        self.profile_sha256 = PINNED_PROFILE_SHA256
        self.manifest_sha256 = _sha256(manifest_path)
        self.engine = PreNMSRuntimeV3(model_dir / _ONNX_NAME)
        self.fingerprint = _canonical_sha256({
            "profile_id": self.profile_id,
            "profile_sha256": self.profile_sha256,
            "onnx_sha256": PINNED_ONNX_SHA256,
            "manifest_sha256": self.manifest_sha256,
        })

    def detect_rgb(self, rgb):
        p = self.profile["preprocessing"]
        tensor, metadata, canvas = preprocess_rgb_v2(rgb, scale=p["scale"], mean=p["mean"],
            channel_order=p["channel_order"], resize_policy=p["resize_policy"], rotation_policy=p["rotation_policy"])
        metadata["RGB_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
        metadata["tensor_sha256"] = hashlib.sha256(tensor.tobytes()).hexdigest()
        metadata["detector_profile_sha256"] = self.profile_sha256
        raw = self.engine.infer_tensor(tensor)
        tail = graph_nms(raw["detection_expand_dim"].reshape(3648, 4),
                         raw["per_box_prob_expand_dim"].reshape(3648, 12), **self.profile["graph_nms"])
        wrapper = self.profile["wrapper"]
        records, rejected = vision_records_v3(tail, wrapper["confidence_floor"])
        thresholds = np.float32(wrapper["class_thresholds"])
        class_pass = []
        for row in records:
            row["detector_crop_normalized_box"] = unpad_and_unrotate(row["model_input_normalized_box"], metadata)
            if np.float32(row["score"]) < thresholds[row["class_index"]]:
                rejected.append(dict(row=row["row"], reason="class_threshold", confidence=row["score"]))
            else:
                class_pass.append(row)
        retained, iom = vision_face_nms_v3(class_pass, wrapper["intersection_over_min_area_threshold"])
        faces, unit_rejected = [], []
        for row in retained:
            if row["class_index"] != 0:
                continue
            square = square_in_detector_crop(row["detector_crop_normalized_box"], metadata)
            source_box, roi_receipt = detector_box_to_source_roi(square, metadata)
            if not source_unit_rect_gate(source_box, semantics=wrapper["unit_rect_gate"]):
                unit_rejected.append(dict(row=row["row"], box=source_box, reason="unit_rect_no_intersection"))
                continue
            faces.append(dict(box=source_box, score=row["score"], label=1, row=row["row"],
                selected_index=row["selected_index"], detector_crop_box=row["detector_crop_normalized_box"],
                squared_detector_crop_box=square, model_input_normalized_box=row["model_input_normalized_box"],
                source_roi_receipt=roi_receipt, box_clipped=False))
        faces.sort(key=lambda face: (-face["score"], face["row"]))
        return dict(faces=faces, metadata=metadata, raw=raw, tail=tail, canvas=canvas,
            wrapper_records=records, wrapper_rejected=rejected, wrapper_class_pass=class_pass,
            wrapper_retained_records=retained, wrapper_IOM=iom, unit_rect_rejected=unit_rejected,
            limits=self.profile["limits"], runtime=self.engine.as_dict())

    def as_dict(self) -> dict[str, object]:
        return dict(producer_id=self.profile_id, producer_fingerprint=self.fingerprint,
                    detector=DETECTOR_ID, detector_onnx_sha256=PINNED_ONNX_SHA256,
                    profile_sha256=self.profile_sha256, manifest_sha256=self.manifest_sha256,
                    runtime_version=self.engine.runtime_version, native_execution=False)


@functools.lru_cache(maxsize=None)
def _detector_for(model_dir: Path) -> Anodv5DetectorV3:
    # A failed build raises and is therefore not cached: the next call checks the files again.
    return Anodv5DetectorV3(model_dir)


def shared_anodv5_detector(model_dir: Path | str | None = None) -> Anodv5DetectorV3:
    """The one detector of this process for a model folder (the checks run on the first call only)."""
    folder = Path(load_settings().anodv5_dir if model_dir is None else model_dir)
    return _detector_for(folder.resolve())
