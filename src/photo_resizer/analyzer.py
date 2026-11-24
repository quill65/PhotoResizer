"""Offline CV/ML scene analysis for smart composition and cropping.

Uses 100% local OpenCV + NumPy algorithms (zero cloud AI calls):
- Multi-cascade face detection (frontal + profile) with headroom/shoulder safety zones
- Spectral Residual + CIE-Lab border contrast + sharpness visual saliency map
- Horizon line detection (Hough segments + vertical gradient continuity)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in normalized [0.0, 1.0] image coordinates."""

    x: float
    y: float
    w: float
    h: float
    score: float = 1.0
    label: str = ""

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w * 0.5

    @property
    def cy(self) -> float:
        return self.y + self.h * 0.5

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def intersection_area(self, other: Box) -> float:
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return (ix2 - ix1) * (iy2 - iy1)

    def iou(self, other: Box) -> float:
        inter = self.intersection_area(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def fraction_inside(self, container: Box) -> float:
        if self.area <= 0:
            return 1.0
        return self.intersection_area(container) / self.area


@dataclass(frozen=True)
class HorizonEstimate:
    """Detected horizon line and relative interest above/below it."""

    y_norm: float | None
    confidence: float
    sky_interest: float = 0.5
    ground_interest: float = 0.5


@dataclass
class SceneAnalysis:
    """Offline scene analysis result used by the smart cropper."""

    faces: list[Box] = field(default_factory=list)
    headroom_boxes: list[Box] = field(default_factory=list)
    subject_box: Box | None = None
    saliency_map: np.ndarray = field(default_factory=lambda: np.ones((1, 1), dtype=np.float32))
    horizon: HorizonEstimate = field(default_factory=lambda: HorizonEstimate(None, 0.0))
    primary_center: tuple[float, float] = (0.5, 0.5)
    proxy_size: tuple[int, int] = (1, 1)


@lru_cache(maxsize=4)
def _load_haar_cascade(filename: str) -> cv2.CascadeClassifier | None:
    cascade_path = Path(cv2.data.haarcascades) / filename
    if not cascade_path.exists():
        return None
    clf = cv2.CascadeClassifier(str(cascade_path))
    if clf.empty():
        return None
    return clf


def _nms_boxes(boxes: list[Box], iou_threshold: float = 0.35) -> list[Box]:
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda b: (b.score, b.area), reverse=True)
    kept: list[Box] = []
    for candidate in ordered:
        if all(candidate.iou(existing) < iou_threshold for existing in kept):
            kept.append(candidate)
    return kept


def _detect_faces(rgb_proxy: np.ndarray) -> list[Box]:
    """Detect frontal and profile faces using bundled offline OpenCV cascades."""
    h, w = rgb_proxy.shape[:2]
    gray = cv2.cvtColor(rgb_proxy, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    min_face = max(24, int(min(w, h) * 0.05))
    raw_boxes: list[Box] = []

    cascade_specs = [
        ("haarcascade_frontalface_alt2.xml", 1.1, 5, 1.0),
        ("haarcascade_frontalface_default.xml", 1.12, 6, 0.85),
        ("haarcascade_profileface.xml", 1.12, 5, 0.75),
    ]

    for filename, scale_factor, min_neighbors, base_score in cascade_specs:
        clf = _load_haar_cascade(filename)
        if clf is None:
            continue
        detections = clf.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_face, min_face),
        )
        for fx, fy, fw, fh in detections:
            raw_boxes.append(
                Box(
                    x=fx / w,
                    y=fy / h,
                    w=fw / w,
                    h=fh / h,
                    score=base_score,
                    label="face",
                )
            )

        # Also detect right-facing profiles by checking the horizontally flipped grayscale image
        if "profile" in filename:
            flipped = cv2.flip(gray, 1)
            f_detections = clf.detectMultiScale(
                flipped,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(min_face, min_face),
            )
            for fx, fy, fw, fh in f_detections:
                orig_x = w - (fx + fw)
                raw_boxes.append(
                    Box(
                        x=orig_x / w,
                        y=fy / h,
                        w=fw / w,
                        h=fh / h,
                        score=base_score,
                        label="face",
                    )
                )

    return _nms_boxes(raw_boxes, iou_threshold=0.35)


def _expand_face_to_headroom(face: Box) -> Box:
    """Create a protective head-and-upper-shoulders box around a face."""
    x1 = max(0.0, face.x - 0.35 * face.w)
    y1 = max(0.0, face.y - 0.55 * face.h)
    x2 = min(1.0, face.x2 + 0.35 * face.w)
    y2 = min(1.0, face.y2 + 0.85 * face.h)
    return Box(x=x1, y=y1, w=x2 - x1, h=y2 - y1, score=face.score, label="headroom")


def _normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if max_val - min_val < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_val) / (max_val - min_val)


def _spectral_residual_saliency(gray: np.ndarray) -> np.ndarray:
    """Hou & Zhang (2007) Spectral Residual saliency using 2D FFT."""
    small_w = 128
    small_h = max(32, int(round(gray.shape[0] * (small_w / max(1, gray.shape[1])))))
    small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA).astype(np.float32)

    fft = np.fft.fft2(small)
    log_amp = np.log(np.abs(fft) + 1e-6)
    phase = np.angle(fft)

    avg_log_amp = cv2.blur(log_amp.astype(np.float32), (3, 3))
    spectral_residual = log_amp - avg_log_amp

    reconstructed = np.fft.ifft2(np.exp(spectral_residual + 1j * phase))
    saliency = np.abs(reconstructed) ** 2
    saliency = cv2.GaussianBlur(saliency.astype(np.float32), (9, 9), 2.5)
    resized = cv2.resize(saliency, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return _normalize_map(resized)


def _lab_border_contrast_saliency(rgb: np.ndarray) -> np.ndarray:
    """Color distinctiveness relative to the 4 image border regions in CIE-Lab.

    Using the minimum distance to each of the 4 border medians prevents a uniform
    sky (matching the top border) from looking salient against a ground-dominated
    bottom border in landscape photographs.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    margin_y = max(2, int(h * 0.06))
    margin_x = max(2, int(w * 0.06))

    border_medians = [
        np.median(lab[:margin_y, :, :].reshape(-1, 3), axis=0),
        np.median(lab[-margin_y:, :, :].reshape(-1, 3), axis=0),
        np.median(lab[:, :margin_x, :].reshape(-1, 3), axis=0),
        np.median(lab[:, -margin_x:, :].reshape(-1, 3), axis=0),
    ]
    dists = [np.linalg.norm(lab - med, axis=2) for med in border_medians]
    dist = np.min(np.stack(dists, axis=0), axis=0)
    dist = cv2.GaussianBlur(dist.astype(np.float32), (11, 11), 0)
    return _normalize_map(dist)


def _sharpness_saliency(gray: np.ndarray) -> np.ndarray:
    """In-focus edge & detail energy (distinguishes sharp subjects from bokeh)."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    blurred = cv2.GaussianBlur(grad_mag, (15, 15), 0)
    return _normalize_map(blurred)


def _compute_saliency_map(
    rgb_proxy: np.ndarray,
    faces: list[Box],
    headroom_boxes: list[Box],
) -> np.ndarray:
    gray = cv2.cvtColor(rgb_proxy, cv2.COLOR_RGB2GRAY)
    spectral = _spectral_residual_saliency(gray)
    color_contrast = _lab_border_contrast_saliency(rgb_proxy)
    sharpness = _sharpness_saliency(gray)

    combined = 0.35 * spectral + 0.40 * color_contrast + 0.25 * sharpness
    combined = _normalize_map(combined)

    # Boost saliency in detected face and headroom zones
    h, w = combined.shape
    if faces:
        boost = np.zeros_like(combined)
        for box in headroom_boxes:
            x1, y1 = int(box.x * w), int(box.y * h)
            x2, y2 = int(np.ceil(box.x2 * w)), int(np.ceil(box.y2 * h))
            boost[y1:y2, x1:x2] = np.maximum(boost[y1:y2, x1:x2], 0.45)
        for face in faces:
            x1, y1 = int(face.x * w), int(face.y * h)
            x2, y2 = int(np.ceil(face.x2 * w)), int(np.ceil(face.y2 * h))
            boost[y1:y2, x1:x2] = np.maximum(boost[y1:y2, x1:x2], 0.95)
        boost = cv2.GaussianBlur(boost, (15, 15), 0)
        combined = np.clip(combined * 0.65 + boost * 0.85, 0.0, 1.0)
        combined = _normalize_map(combined)

    return combined


def _extract_subject_box(saliency: np.ndarray, faces: list[Box]) -> Box | None:
    """Find the bounding box covering the primary salient subject."""
    h, w = saliency.shape
    threshold = max(0.35, float(np.percentile(saliency, 80)))
    mask = (saliency >= threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    best_box: Box | None = None
    best_mass = 0.0
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < max(16, int(w * h * 0.005)):
            continue
        mass = float(saliency[labels == idx].sum())
        if mass > best_mass:
            best_mass = mass
            bx = int(stats[idx, cv2.CC_STAT_LEFT])
            by = int(stats[idx, cv2.CC_STAT_TOP])
            bw = int(stats[idx, cv2.CC_STAT_WIDTH])
            bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
            best_box = Box(bx / w, by / h, bw / w, bh / h, score=1.0, label="subject")

    if faces and best_box is not None:
        # Union primary salient subject with face union so subjects stay intact
        min_x = min([best_box.x] + [f.x for f in faces])
        min_y = min([best_box.y] + [f.y for f in faces])
        max_x = max([best_box.x2] + [f.x2 for f in faces])
        max_y = max([best_box.y2] + [f.y2 for f in faces])
        return Box(min_x, min_y, max_x - min_x, max_y - min_y, score=1.0, label="subject")
    return best_box


def _detect_horizon(
    rgb_proxy: np.ndarray,
    saliency: np.ndarray,
    faces: list[Box],
) -> HorizonEstimate:
    """Estimate dominant horizontal horizon line using edges + row-wise vertical gradients."""
    # Close-up portraits shouldn't be steered by background horizontal lines
    total_face_area = sum(f.area for f in faces)
    if total_face_area > 0.08:
        return HorizonEstimate(y_norm=None, confidence=0.0)

    h, w = rgb_proxy.shape[:2]
    if h < 40 or w < 40:
        return HorizonEstimate(y_norm=None, confidence=0.0)

    gray = cv2.cvtColor(rgb_proxy, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Hough near-horizontal line voting across rows
    edges = cv2.Canny(blurred, 40, 120)
    min_line_len = max(20, int(w * 0.22))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(25, int(w * 0.12)),
        minLineLength=min_line_len,
        maxLineGap=max(10, int(w * 0.08)),
    )

    hough_profile = np.zeros(h, dtype=np.float32)
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            dx = abs(int(x2) - int(x1))
            dy = abs(int(y2) - int(y1))
            if dx < 1:
                continue
            angle_deg = float(np.degrees(np.arctan2(dy, dx)))
            if angle_deg <= 7.0:
                length = float(np.hypot(dx, dy))
                mid_y = int(round((y1 + y2) * 0.5))
                if 0 <= mid_y < h:
                    hough_profile[mid_y] += length / w

    # 2. Row-wise vertical gradient continuity (strong across-row transition, e.g., sky/land)
    lab = cv2.cvtColor(rgb_proxy, cv2.COLOR_RGB2LAB).astype(np.float32)
    row_means = lab.mean(axis=1)  # (h, 3)
    # Compare average color in a thin band above vs below each row
    band = max(3, int(h * 0.02))
    color_step = np.zeros(h, dtype=np.float32)
    for y in range(band, h - band):
        above = row_means[y - band : y].mean(axis=0)
        below = row_means[y : y + band].mean(axis=0)
        color_step[y] = float(np.linalg.norm(above - below))

    # Smooth both profiles vertically
    kernel_size = max(5, (int(h * 0.03) | 1))
    hough_smooth = cv2.GaussianBlur(hough_profile.reshape(-1, 1), (1, kernel_size), 0).ravel()
    step_smooth = cv2.GaussianBlur(color_step.reshape(-1, 1), (1, kernel_size), 0).ravel()

    # Restrict search band to [12%, 88%] to ignore borders
    y_start = int(h * 0.12)
    y_end = int(h * 0.88)

    hough_norm = np.clip(hough_smooth / 0.8, 0.0, 1.0)
    step_norm = np.clip(step_smooth / 40.0, 0.0, 1.0)
    combined_profile = 0.60 * hough_norm + 0.40 * step_norm

    band_scores = combined_profile[y_start:y_end]
    if band_scores.size == 0:
        return HorizonEstimate(y_norm=None, confidence=0.0)

    best_rel = int(np.argmax(band_scores))
    best_y = y_start + best_rel
    peak_val = float(band_scores[best_rel])
    median_val = float(np.median(band_scores)) + 1e-5
    prominence = peak_val / median_val

    # Require both decent absolute signal and prominence over background rows
    confidence = float(np.clip(peak_val * min(1.0, (prominence - 1.1) / 2.0), 0.0, 1.0))
    if confidence < 0.25:
        return HorizonEstimate(y_norm=None, confidence=confidence)

    y_norm = best_y / h
    # Measure actual visual detail & color richness above vs below the horizon
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    pad = max(2, int(h * 0.02))
    sky_end = max(1, best_y - pad)
    gnd_start = min(h - 1, best_y + pad)

    sky_detail = float(grad_mag[:sky_end, :].mean()) / 255.0
    sky_var = float(lab[:sky_end, :, :].std(axis=(0, 1)).mean()) / 64.0
    gnd_detail = float(grad_mag[gnd_start:, :].mean()) / 255.0
    gnd_var = float(lab[gnd_start:, :, :].std(axis=(0, 1)).mean()) / 64.0

    sky_interest = 0.55 * sky_detail + 0.45 * sky_var
    ground_interest = 0.55 * gnd_detail + 0.45 * gnd_var
    return HorizonEstimate(
        y_norm=y_norm,
        confidence=confidence,
        sky_interest=sky_interest,
        ground_interest=ground_interest,
    )


def analyze_scene(rgb_image: np.ndarray, max_dim: int = 640) -> SceneAnalysis:
    """Run offline CV/ML scene analysis on an RGB uint8 numpy image."""
    orig_h, orig_w = rgb_image.shape[:2]
    scale = min(1.0, max_dim / max(orig_w, orig_h))
    if scale < 1.0:
        proxy_w = max(16, int(round(orig_w * scale)))
        proxy_h = max(16, int(round(orig_h * scale)))
        proxy = cv2.resize(rgb_image, (proxy_w, proxy_h), interpolation=cv2.INTER_AREA)
    else:
        proxy = rgb_image
        proxy_h, proxy_w = orig_h, orig_w

    faces = _detect_faces(proxy)
    headroom_boxes = [_expand_face_to_headroom(face) for face in faces]
    saliency = _compute_saliency_map(proxy, faces, headroom_boxes)
    subject_box = _extract_subject_box(saliency, faces)
    horizon = _detect_horizon(proxy, saliency, faces)

    # Saliency center of mass
    total_mass = float(saliency.sum())
    if total_mass > 1e-6:
        ys, xs = np.indices((proxy_h, proxy_w), dtype=np.float32)
        cx = float((saliency * xs).sum() / total_mass) / proxy_w
        cy = float((saliency * ys).sum() / total_mass) / proxy_h
    else:
        cx, cy = 0.5, 0.5

    return SceneAnalysis(
        faces=faces,
        headroom_boxes=headroom_boxes,
        subject_box=subject_box,
        saliency_map=saliency,
        horizon=horizon,
        primary_center=(cx, cy),
        proxy_size=(proxy_w, proxy_h),
    )
