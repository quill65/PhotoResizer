"""Offline neural wildlife & foreground subject detection using ONNX Runtime.

Combines two lightweight local ONNX models (100% offline after one-time download,
zero cloud AI APIs):
1. U2-Net-P (`u2netp.onnx`, ~4.6 MB): Salient foreground object segmentation that
   isolates *any* animal or subject (insects, spiders, birds, squirrels, deer, frogs,
   mammals, flowers) from busy foliage, branches, grass, or bokeh backgrounds.
2. YOLO11n / YOLOv8n (`yolo11n.onnx`, ~11 MB): Fast COCO object detector for
   precise bounding boxes of birds, mammals (cat/dog/horse/sheep/cow/elephant/bear/
   zebra/giraffe), and people — especially helpful for small perched birds in trees.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import httpx
import numpy as np
import onnxruntime as ort

U2NETP_FILENAME = "u2netp.onnx"
U2NETP_URLS: tuple[str, ...] = (
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
    "https://huggingface.co/nfa04/u2netp-onnx/resolve/main/u2netp.onnx",
)

YOLO_FILENAME = "yolo11n.onnx"
YOLO_URLS: tuple[str, ...] = (
    "https://huggingface.co/piky/yolo11/resolve/59fe0d02e80c78849132f90e2b93d252b02c61cf/yolo11n.onnx",
    "https://huggingface.co/unity/inference-engine-yolo/resolve/6c8bec2eab5fef018527d76285c92b18ac4fd0aa/models/yolo11n.onnx",
    "https://huggingface.co/unity/inference-engine-yolo/resolve/ed7f4daf9263d0d31be1d60b9d67c8baea721d60/yolov8n.onnx",
)

# COCO class IDs relevant to wildlife & portrait subjects
COCO_ANIMAL_CLASSES: dict[int, str] = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}
COCO_PERSON_CLASS = 0


@dataclass(frozen=True)
class DetectedSubject:
    """Normalized [0, 1] detection box for an animal or primary foreground subject."""

    x: float
    y: float
    w: float
    h: float
    score: float
    label: str


@dataclass
class NeuralSubjectResult:
    """Combined output from local neural foreground & animal detectors."""

    animals: list[DetectedSubject] = field(default_factory=list)
    people: list[DetectedSubject] = field(default_factory=list)
    foreground_mask: np.ndarray | None = None  # float32 (H, W) in [0, 1]
    detector_tags: list[str] = field(default_factory=list)


def _detector_cache_dir() -> Path:
    base = os.environ.get("PHOTO_RESIZER_CACHE_DIR")
    if base:
        return Path(base) / "models" / "detectors"
    return Path.home() / ".cache" / "photo_resizer" / "models" / "detectors"


def _find_detector_file(filename: str) -> Path | None:
    search_dirs = [
        Path.cwd() / "models" / "detectors",
        Path.cwd() / "models",
        Path(__file__).resolve().parents[2] / "models" / "detectors",
        Path(__file__).resolve().parents[2] / "models",
        _detector_cache_dir(),
    ]
    for directory in search_dirs:
        candidate = directory / filename
        if candidate.is_file() and candidate.stat().st_size > 100_000:
            return candidate
    return None


def _download_model(filename: str, urls: tuple[str, ...], timeout: float = 35.0) -> Path | None:
    existing = _find_detector_file(filename)
    if existing is not None:
        return existing

    dest_dir = _detector_cache_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    for url in urls:
        tmp_path: Path | None = None
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as resp:
                resp.raise_for_status()
                with tempfile.NamedTemporaryFile(
                    dir=dest_dir, prefix="dl_", suffix=".onnx.tmp", delete=False
                ) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    for chunk in resp.iter_bytes(chunk_size=1024 * 256):
                        tmp_file.write(chunk)
            if tmp_path is not None and tmp_path.stat().st_size > 100_000:
                tmp_path.replace(dest_path)
                return dest_path
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
        except Exception:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
    return None


def _create_ort_session(model_path: Path) -> ort.InferenceSession:
    available = ort.get_available_providers()
    preferred = ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [p for p in preferred if p in available] or ["CPUExecutionProvider"]

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)


def _nms_subjects(items: list[DetectedSubject], iou_thresh: float = 0.45) -> list[DetectedSubject]:
    if not items:
        return []
    ordered = sorted(items, key=lambda d: d.score, reverse=True)
    kept: list[DetectedSubject] = []
    for cand in ordered:
        cx2, cy2 = cand.x + cand.w, cand.y + cand.h
        c_area = max(0.0, cand.w) * max(0.0, cand.h)
        overlap = False
        for prev in kept:
            px2, py2 = prev.x + prev.w, prev.y + prev.h
            ix1, iy1 = max(cand.x, prev.x), max(cand.y, prev.y)
            ix2, iy2 = min(cx2, px2), min(cy2, py2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                p_area = max(0.0, prev.w) * max(0.0, prev.h)
                union = c_area + p_area - inter
                if union > 0 and (inter / union) >= iou_thresh:
                    overlap = True
                    break
        if not overlap:
            kept.append(cand)
    return kept


class _U2NetSegmenter:
    """Lightweight U2-Net-P salient foreground segmenter (birds, insects, mammals, etc.)."""

    def __init__(self, session: ort.InferenceSession) -> None:
        self._session = session
        inp = session.get_inputs()[0]
        self._input_name = inp.name
        self._output_name = session.get_outputs()[0].name
        shape = inp.shape
        self._in_h = shape[2] if len(shape) >= 4 and isinstance(shape[2], int) else 320
        self._in_w = shape[3] if len(shape) >= 4 and isinstance(shape[3], int) else 320

    def segment(self, rgb_image: np.ndarray, out_size: tuple[int, int]) -> np.ndarray | None:
        # Uniform/blank images have no distinct foreground subject to segment
        if float(rgb_image.std()) < 3.0:
            return None

        resized = cv2.resize(
            rgb_image, (self._in_w, self._in_h), interpolation=cv2.INTER_AREA
        ).astype(np.float32)

        # Standard U2-Net ImageNet normalization
        max_val = float(resized.max())
        if max_val > 1e-6:
            resized /= max_val
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (resized - mean) / std
        nchw = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        raw = self._session.run([self._output_name], {self._input_name: nchw})[0]
        pred = np.squeeze(raw).astype(np.float32)
        if pred.ndim != 2:
            return None

        p_min, p_max = float(pred.min()), float(pred.max())
        # Ignore empty/near-flat predictions (no clear foreground subject)
        if p_max < 0.25 or (p_max - p_min) < 0.15:
            return None

        mask = (pred - p_min) / (p_max - p_min)
        # Suppress low-confidence background halo so foreground animals stand out cleanly
        mask = np.where(mask >= 0.20, mask, 0.0).astype(np.float32)
        fg_coverage = float((mask >= 0.30).mean())
        if float(mask.max()) < 0.30 or fg_coverage > 0.85:
            return None

        out_w, out_h = out_size
        return cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


class _YoloAnimalDetector:
    """YOLO11n / YOLOv8n COCO detector focused on birds, mammals, and people."""

    def __init__(self, session: ort.InferenceSession) -> None:
        self._session = session
        inp = session.get_inputs()[0]
        self._input_name = inp.name
        self._output_name = session.get_outputs()[0].name
        shape = inp.shape
        self._in_h = shape[2] if len(shape) >= 4 and isinstance(shape[2], int) else 640
        self._in_w = shape[3] if len(shape) >= 4 and isinstance(shape[3], int) else 640

    def detect(
        self,
        rgb_image: np.ndarray,
        animal_conf: float = 0.22,
        person_conf: float = 0.35,
    ) -> tuple[list[DetectedSubject], list[DetectedSubject]]:
        orig_h, orig_w = rgb_image.shape[:2]
        scale = min(self._in_w / orig_w, self._in_h / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        pad_x = (self._in_w - new_w) // 2
        pad_y = (self._in_h - new_h) // 2

        resized = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self._in_h, self._in_w, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        nchw = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]
        raw = self._session.run([self._output_name], {self._input_name: nchw})[0]

        preds = np.squeeze(raw, axis=0)  # (84, 8400) or (8400, 84)
        if preds.ndim != 2:
            return [], []
        if preds.shape[0] in (84, 85) and preds.shape[1] not in (84, 85):
            preds = preds.T  # -> (N, 4 + num_classes)
        elif preds.shape[0] < preds.shape[1]:
            preds = preds.T
        if preds.shape[1] < 4 + 24:
            return [], []

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        animals: list[DetectedSubject] = []
        people: list[DetectedSubject] = []

        # Filter relevant classes quickly with numpy mask
        animal_ids = np.array(list(COCO_ANIMAL_CLASSES.keys()), dtype=np.int64)
        is_animal = np.isin(class_ids, animal_ids) & (confidences >= animal_conf)
        is_person = (class_ids == COCO_PERSON_CLASS) & (confidences >= person_conf)
        keep_indices = np.nonzero(is_animal | is_person)[0]

        for idx in keep_indices:
            cid = int(class_ids[idx])
            conf = float(confidences[idx])
            cx, cy, bw, bh = boxes_xywh[idx]

            # Handle models that output normalized [0..1] coords vs pixel coords
            if max(float(cx), float(cy), float(bw), float(bh)) <= 2.0:
                cx *= self._in_w
                cy *= self._in_h
                bw *= self._in_w
                bh *= self._in_h

            x1 = ((float(cx) - float(bw) * 0.5) - pad_x) / max(scale, 1e-6)
            y1 = ((float(cy) - float(bh) * 0.5) - pad_y) / max(scale, 1e-6)
            x2 = ((float(cx) + float(bw) * 0.5) - pad_x) / max(scale, 1e-6)
            y2 = ((float(cy) + float(bh) * 0.5) - pad_y) / max(scale, 1e-6)

            nx1 = max(0.0, min(1.0, x1 / orig_w))
            ny1 = max(0.0, min(1.0, y1 / orig_h))
            nx2 = max(0.0, min(1.0, x2 / orig_w))
            ny2 = max(0.0, min(1.0, y2 / orig_h))
            if nx2 - nx1 < 0.015 or ny2 - ny1 < 0.015:
                continue

            if cid in COCO_ANIMAL_CLASSES:
                animals.append(
                    DetectedSubject(
                        x=nx1,
                        y=ny1,
                        w=nx2 - nx1,
                        h=ny2 - ny1,
                        score=conf,
                        label=COCO_ANIMAL_CLASSES[cid],
                    )
                )
            elif cid == COCO_PERSON_CLASS:
                people.append(
                    DetectedSubject(
                        x=nx1,
                        y=ny1,
                        w=nx2 - nx1,
                        h=ny2 - ny1,
                        score=conf,
                        label="person",
                    )
                )

        return _nms_subjects(animals), _nms_subjects(people)


class WildlifeSubjectDetector:
    """Combined offline neural detector (U2-Net-P foreground + YOLO11n animals)."""

    def __init__(self, auto_download: bool = True) -> None:
        self._auto_download = auto_download
        self._initialized = False
        self._u2net: _U2NetSegmenter | None = None
        self._yolo: _YoloAnimalDetector | None = None

    def _ensure_loaded(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        u2_path = (
            _download_model(U2NETP_FILENAME, U2NETP_URLS)
            if self._auto_download
            else _find_detector_file(U2NETP_FILENAME)
        )
        if u2_path is not None:
            try:
                self._u2net = _U2NetSegmenter(_create_ort_session(u2_path))
            except Exception:
                self._u2net = None

        yolo_path = (
            _download_model(YOLO_FILENAME, YOLO_URLS)
            if self._auto_download
            else _find_detector_file(YOLO_FILENAME)
        )
        if yolo_path is not None:
            try:
                self._yolo = _YoloAnimalDetector(_create_ort_session(yolo_path))
            except Exception:
                self._yolo = None

    def detect(
        self,
        rgb_image: np.ndarray,
        proxy_size: tuple[int, int],
    ) -> NeuralSubjectResult:
        self._ensure_loaded()
        result = NeuralSubjectResult()

        if self._yolo is not None:
            try:
                animals, people = self._yolo.detect(rgb_image)
                result.animals = animals
                result.people = people
                if animals:
                    labels = sorted({a.label for a in animals})
                    result.detector_tags.append(f"yolo11n:{'+'.join(labels)}")
                elif people:
                    result.detector_tags.append("yolo11n:person")
            except Exception:
                pass

        if self._u2net is not None:
            try:
                mask = self._u2net.segment(rgb_image, proxy_size)
                if mask is not None:
                    result.foreground_mask = mask
                    result.detector_tags.append("u2netp_foreground")
            except Exception:
                pass

        return result


@lru_cache(maxsize=2)
def get_wildlife_detector(auto_download: bool = True) -> WildlifeSubjectDetector:
    """Return a cached WildlifeSubjectDetector so ONNX sessions load only once per process."""
    return WildlifeSubjectDetector(auto_download=auto_download)
