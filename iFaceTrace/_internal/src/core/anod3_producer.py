"""Validated ANODv3 detector + direct-BGRA fp3.1 recognition producer."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from threadpoolctl import threadpool_limits

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - optional outside packaged builds
    pillow_heif = None  # type: ignore

from src.config import SUBPROCESS_NO_WINDOW

_RUNTIME = Path(__file__).with_name("anod3_runtime")
_PROFILE = _RUNTIME / "profile.json"
RECOGNITION_CROP_ID = "partial-placement-ios16-static-host-v1"
_PROFILE_ID = "anod3-bilinear-owned97-outer107-inner129-fresh-winner-nms-vdsp-order-v2"
_HEIF_SUFFIXES = {".heic", ".heif", ".hif"}
_ARTIFACT_NAMES = {"anod3_net", "anod3_weights", "facerec_net", "facerec_weights", "anchors"}
_RUNTIME_NAMES = {"../anod3_producer.py", "__init__.py", "anod3_inference.py",
                  "detector_preprocess.py", "engine.py", "espresso_inference.py",
                  "face_postprocess.py"}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def integral(rect):
    x, y, w, h = rect
    assert w > 0 and h > 0 and all(map(math.isfinite, rect))
    left, bottom = math.floor(x), math.floor(y)
    return left, bottom, math.ceil(x+w)-left, math.ceil(y+h)-bottom


def intersection(rect, width, height):
    x, y, w, h = rect
    left, bottom = max(0, x), max(0, y)
    right, top = min(width, x+w), min(height, y+h)
    assert right > left and top > bottom, 'Only nonempty intersections are supported'
    return left, bottom, right-left, top-bottom


def cv_destination(rect, source_width, source_height, dst_width=128, dst_height=128):
    """0x1a0b791a4..1d0 then 0x1a0b79918/64, orientation1 partial path.

    Input is Vision lower-left ROI. Returned destination is already top-left;
    do not flip it again for PIL. No YUV alignment or backend kernel is implied.
    """
    x, y, w, h = rect
    assert w > 0 and h > 0
    clipped = intersection(rect, source_width, source_height)
    cx, cy, cw, ch = clipped
    src_top_left = (cx, source_height-(cy+ch), cw, ch)
    if clipped == rect:
        return dict(clipped=clipped, source_top_left=src_top_left,
                    destination=(0, 0, dst_width, dst_height))
    dx = max(-x, 0)*(dst_width/w)
    dy = (h-max(-y, 0)-ch)*(dst_height/h)
    floating = (dx, dy, cw*(dst_width/w), ch*(dst_height/h))
    dest = intersection(integral(floating), dst_width, dst_height)
    return dict(clipped=clipped, source_top_left=src_top_left,
                floating=floating, destination=dest)


class Anod3Producer:
    """One pinned producer. Any code/model/CLI mismatch is a hard failure."""

    def __init__(self, *, models_dir: Path, anchors: Path, faceprint_cli: Path,
                 profile_path: Path = _PROFILE) -> None:
        self.models_dir = Path(models_dir)
        self.anchors = Path(anchors)
        self.faceprint_cli = Path(faceprint_cli)
        self.profile_path = Path(profile_path)
        profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
        if profile.get("profile_id") != _PROFILE_ID:
            raise ValueError("wrong ANODv3 producer profile")
        if set(profile.get("artifact_sha256", {})) != _ARTIFACT_NAMES:
            raise ValueError("incomplete ANODv3 artifact contract")
        if set(profile.get("runtime_sha256", {})) != _RUNTIME_NAMES:
            raise ValueError("incomplete ANODv3 runtime contract")
        if set(profile.get("faceprint_dll_sha256", {})) != {"heif.dll", "libde265.dll"}:
            raise ValueError("incomplete ANODv3 decoder DLL contract")

        artifacts = {
            "anod3_net": self.models_dir / "anod_v3_d3/model_opt.espresso.net",
            "anod3_weights": self.models_dir / "anod_v3_d3/model_opt.espresso.weights",
            "facerec_net": self.models_dir / "facerec_fp3.1_fa1.3.espresso.net",
            "facerec_weights": self.models_dir / "facerec_fp3.1_fa1.3.espresso.weights_nonane",
            "anchors": self.anchors,
        }
        runtime = {name: (_RUNTIME / name).resolve() for name in profile["runtime_sha256"]}
        if any(not path.is_relative_to(_RUNTIME.parent.resolve()) for path in runtime.values()):
            raise ValueError("ANODv3 profile runtime path escapes src/core")
        for name, path in {**artifacts, **runtime}.items():
            expected = profile[("artifact_sha256" if name in artifacts else "runtime_sha256")][name]
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"ANODv3 profile file mismatch: {name}")
        cli_sha256 = _sha256(self.faceprint_cli) if self.faceprint_cli.is_file() else ""
        if cli_sha256 not in profile["faceprint_cli_sha256"]:
            raise ValueError("ANODv3 profile faceprint CLI mismatch")
        dlls = {name: self.faceprint_cli.parent / name for name in profile["faceprint_dll_sha256"]}
        for name, path in dlls.items():
            if not path.is_file() or _sha256(path) != profile["faceprint_dll_sha256"][name]:
                raise ValueError(f"ANODv3 profile decoder DLL mismatch: {name}")

        self.profile_id = profile["profile_id"]
        self.profile_sha256 = _sha256(self.profile_path)
        self.cli_sha256 = cli_sha256
        self.model_sha256 = _canonical_sha256(
            {name: profile["artifact_sha256"][name] for name in sorted(artifacts)}
        )
        self.source_manifest_sha256 = profile["frozen_source_manifest_sha256"]
        self.fingerprint = _canonical_sha256({
            "producer_id": self.profile_id,
            "profile_sha256": self.profile_sha256,
            "model_sha256": self.model_sha256,
            "cli_sha256": self.cli_sha256,
            "faceprint_dll_sha256": profile["faceprint_dll_sha256"],
            "source_manifest_sha256": self.source_manifest_sha256,
        })
        work_sources = {**artifacts, "faceprint_cli": self.faceprint_cli, **dlls}
        source_state = {
            name: (path.stat().st_size, path.stat().st_mtime_ns, _sha256(path))
            for name, path in work_sources.items()
        }
        self._work = tempfile.TemporaryDirectory(prefix="face_analysis_anod3_profile_")
        work = Path(self._work.name)
        copied: dict[str, Path] = {}
        for name, path in work_sources.items():
            destination = work / (name if name in dlls else f"{name}{path.suffix}")
            shutil.copy2(path, destination)
            copied[name] = destination
            if _sha256(destination) != source_state[name][2]:
                raise RuntimeError(f"ANODv3 work-copy mismatch: {name}")
        if source_state != {
            name: (path.stat().st_size, path.stat().st_mtime_ns, _sha256(path))
            for name, path in work_sources.items()
        }:
            raise RuntimeError("ANODv3 profile source changed while copying")
        from src.core.anod3_runtime.detector_preprocess import preprocess_rgb
        from src.core.anod3_runtime.engine import ANOD3ModelNMS

        self.faceprint_cli = copied["faceprint_cli"]
        self._preprocess_rgb = preprocess_rgb
        self._engine = ANOD3ModelNMS(
            copied["anod3_net"], copied["anod3_weights"], copied["anchors"],
            self.profile_sha256,
        )
        self._facerec_net = copied["facerec_net"]
        self._facerec_weights = copied["facerec_weights"]
        self._detector_lock = threading.Lock()

    @staticmethod
    def load_rgb(image_path: Path) -> np.ndarray:
        if image_path.suffix.lower() in _HEIF_SUFFIXES and pillow_heif is not None:
            image = pillow_heif.open_heif(str(image_path), convert_hdr_to_8bit=True).to_pillow()
        else:
            image = ImageOps.exif_transpose(Image.open(image_path))
        return np.asarray(image.convert("RGB"))

    def detect(self, image_path: Path) -> tuple[np.ndarray, dict]:
        rgb = self.load_rgb(Path(image_path))
        tensor, meta, _canvas = self._preprocess_rgb(rgb)
        # The interpreter owns mutable NMS diagnostics; one instance is serialized.
        with self._detector_lock, threadpool_limits(limits=1, user_api="blas"):
            result = self._engine.infer_faces(tensor, meta)
        return rgb, result

    @staticmethod
    def recognition_input(
        rgb: np.ndarray, v6: dict[str, float], *, metadata: dict | None = None,
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """Return the product-compatible BGRA/ROI pair; optionally record geometry.

        sourceRect is the un-clipped Vision lower-left (x, y, width, height).
        clippedROI is the product's Pillow top-left (left, top, right, bottom).
        destinationRect is top-left (x, y, width, height), with no second Y flip.
        """
        if (not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3
                or rgb.dtype != np.uint8 or min(rgb.shape[:2]) <= 0):
            raise ValueError("expected nonempty uint8 RGB image")
        h, w = rgb.shape[:2]
        x, y, width, height = (float(v6[key]) for key in ("x", "y", "w", "h"))
        if not np.isfinite([x, y, width, height]).all() or min(width, height) <= 0:
            raise ValueError("invalid face rectangle")

        # Keep the product's FP32 inset and FP64 operation order, including x1/y1.
        # expanded_roi() groups the endpoint additions differently at rounding ties.
        inset = float(np.float32(np.float32(1.765) - np.float32(1)))
        px, py, pw, ph = x * w, y * h, width * w, height * h
        dx, dy = -pw * inset * .5, -ph * inset * .5
        x0, y0 = px + dx, py + dy
        x1, y1 = x0 + pw - 2 * dx, y0 + ph - 2 * dy
        roi = (max(0, math.floor(x0)), max(0, h - math.ceil(y1)),
               min(w, math.ceil(x1)), min(h, h - math.floor(y0)))
        if roi[2] <= roi[0] or roi[3] <= roi[1]:
            raise ValueError("face rectangle outside image")
        source_rect = (math.floor(x0), math.floor(y0),
                       math.ceil(x1) - math.floor(x0), math.ceil(y1) - math.floor(y0))
        geometry = cv_destination(source_rect, w, h)
        left, top, dw, dh = geometry["destination"]

        crop = Image.fromarray(rgb).crop(roi).resize((dw, dh), Image.Resampling.LANCZOS)
        crop_rgb = np.asarray(crop)
        bgra = np.zeros((128, 128, 4), dtype=np.uint8)
        bgra[top:top + dh, left:left + dw, :3] = crop_rgb[:, :, ::-1]
        bgra[top:top + dh, left:left + dw, 3] = 255
        if metadata is not None:
            metadata.update(sourceRect=source_rect, clippedROI=roi,
                            destinationRect=geometry["destination"],
                            intersection_lower_left=geometry["clipped"],
                            destination_floating=geometry.get("floating"),
                            outside_source=source_rect != geometry["clipped"],
                            geometry_scope="iOS16-static-orientation1-host-candidate")
        return bgra, roi

    def embed_bgra(self, bgra: np.ndarray, work_dir: Path) -> np.ndarray:
        if bgra.shape != (128, 128, 4) or bgra.dtype != np.uint8:
            raise ValueError("expected uint8 BGRA 128x128")
        work_dir.mkdir(parents=True, exist_ok=True)
        raw = work_dir / "face.bgra"
        output = work_dir / "face.f32"
        invalid_image = work_dir / "unused.heic"
        invalid_v6 = work_dir / "unused.json"
        raw.write_bytes(bgra.tobytes())
        output.unlink(missing_ok=True)
        invalid_image.write_bytes(b"BGRA input bypasses image decode\n")
        invalid_v6.write_text("{}", encoding="utf-8")
        proc = subprocess.run(
            [str(self.faceprint_cli), str(invalid_image), str(invalid_v6),
             "--bgra-input", str(raw), "--mode", "embedding",
             "--net", str(self._facerec_net), "--weights", str(self._facerec_weights),
             "--out", str(output), "--bgra-input-skip-transpose"],
            capture_output=True, text=True, **SUBPROCESS_NO_WINDOW,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"faceprint_cli failed (exit {proc.returncode}): {proc.stderr[-400:]}")
        vector = np.fromfile(output, dtype="<f4").astype(np.float32).copy()
        if vector.shape != (128,) or not np.isfinite(vector).all():
            raise RuntimeError(f"unexpected faceprint vector: {vector.shape}")
        norm = float(np.linalg.norm(vector.astype(np.float64)))
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-5:
            raise RuntimeError(f"unexpected faceprint norm: {norm}")
        return vector

    def vectorize_image(self, image_path: Path) -> list[dict]:
        rgb, result = self.detect(Path(image_path))
        rows: list[dict] = []
        with tempfile.TemporaryDirectory(prefix="face_analysis_anod3_") as tmp_name:
            tmp = Path(tmp_name)
            for index, face in enumerate(result["faces"]):
                v6 = dict(zip(("x", "y", "w", "h"), map(float, face["box"])))
                bgra, roi = self.recognition_input(rgb, v6)
                vector = self.embed_bgra(bgra, tmp / f"face-{index}")
                rows.append({"vector": vector, "v6_rect": v6, "score": float(face["score"]),
                             "label": 1, "roi": roi})
        return rows

    def as_dict(self) -> dict[str, str]:
        return {
            "recognition_crop": RECOGNITION_CROP_ID,
            "producer_id": self.profile_id,
            "producer_fingerprint": self.fingerprint,
            "model_sha256": self.model_sha256,
            "cli_sha256": self.cli_sha256,
            "profile_sha256": self.profile_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
        }
