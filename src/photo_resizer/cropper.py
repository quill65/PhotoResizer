"""Smart aspect-ratio cropper driven by offline scene analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from photo_resizer.analyzer import Box, SceneAnalysis, analyze_scene
from photo_resizer.config import CropConfig


@dataclass(frozen=True)
class CropBox:
    """Exact pixel crop rectangle (left, top, width, height)."""

    x: int
    y: int
    width: int
    height: int

    @property
    def as_pil_box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class CropDecision:
    """Chosen crop rectangle along with human-readable explanation."""

    box: CropBox
    was_cropped: bool
    score: float
    reason: str
    analysis: SceneAnalysis | None = None


def load_oriented_image(path: Path) -> tuple[Image.Image, bytes | None]:
    """Load an image with EXIF auto-orientation applied and safe EXIF preserved."""
    with Image.open(path) as img:
        raw_exif = img.info.get("exif")
        oriented = ImageOps.exif_transpose(img)
        if oriented.mode not in ("RGB", "RGBA"):
            oriented = oriented.convert("RGB")
        else:
            oriented = oriented.copy()

    # Reset orientation tag in preserved EXIF to 1 (Normal) so viewers won't double-rotate
    cleaned_exif: bytes | None = None
    if raw_exif:
        try:
            exif_obj = Image.Exif()
            exif_obj.load(raw_exif)
            exif_obj[274] = 1  # Orientation = Horizontal (normal)
            cleaned_exif = exif_obj.tobytes()
        except Exception:
            cleaned_exif = None
    return oriented, cleaned_exif


def _maximal_crop_dimensions(src_w: int, src_h: int, target_ratio: float) -> tuple[int, int]:
    """Largest integer (width, height) inside (src_w, src_h) matching target_ratio."""
    src_ratio = src_w / src_h
    if src_ratio > target_ratio:
        # Source is wider -> crop horizontally
        crop_h = src_h
        crop_w = max(1, min(src_w, int(round(src_h * target_ratio))))
    else:
        # Source is taller -> crop vertically
        crop_w = src_w
        crop_h = max(1, min(src_h, int(round(src_w / target_ratio))))
    return crop_w, crop_h


def _rect_sum(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    """O(1) rectangle sum using OpenCV integral image."""
    h, w = integral.shape[0] - 1, integral.shape[1] - 1
    x1 = max(0, min(w, x1))
    x2 = max(x1, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(y1, min(h, y2))
    return float(integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])


def _score_candidate(
    cand: Box,
    analysis: SceneAnalysis,
    integral: np.ndarray,
    total_saliency: float,
    img_w: int,
    img_h: int,
) -> float:
    """Score a candidate normalized crop box for visual composition quality."""
    proxy_w, proxy_h = analysis.proxy_size
    px1 = int(round(cand.x * proxy_w))
    py1 = int(round(cand.y * proxy_h))
    px2 = int(round(cand.x2 * proxy_w))
    py2 = int(round(cand.y2 * proxy_h))

    # 1. Saliency capture fraction (0..1)
    captured = _rect_sum(integral, px1, py1, px2, py2)
    saliency_score = captured / max(total_saliency, 1e-6)

    # 2. Face & headroom protection (heavily penalize clipping faces or top of heads)
    face_penalty = 0.0
    for face in analysis.faces:
        inside = face.fraction_inside(cand)
        if inside < 0.999:
            face_penalty += 4.5 * (1.0 - inside)
    for headroom in analysis.headroom_boxes:
        inside = headroom.fraction_inside(cand)
        if inside < 0.999:
            face_penalty += 1.6 * (1.0 - inside)

    # 3. Subject edge-slicing penalty: penalize cuts passing through dense saliency
    border_penalty = 0.0
    strip = max(2, int(round(min(proxy_w, proxy_h) * 0.025)))
    # Check each edge that actually cuts into the image (not touching outer frame)
    if cand.x > 0.005:
        cut_band = _rect_sum(integral, px1 - strip, py1, px1 + strip, py2)
        cut_area = max(1, (min(proxy_w, px1 + strip) - max(0, px1 - strip)) * max(1, py2 - py1))
        border_penalty += (cut_band / cut_area) * 0.9
    if cand.x2 < 0.995:
        cut_band = _rect_sum(integral, px2 - strip, py1, px2 + strip, py2)
        cut_area = max(1, (min(proxy_w, px2 + strip) - max(0, px2 - strip)) * max(1, py2 - py1))
        border_penalty += (cut_band / cut_area) * 0.9
    if cand.y > 0.005:
        cut_band = _rect_sum(integral, px1, py1 - strip, px2, py1 + strip)
        cut_area = max(1, max(1, px2 - px1) * (min(proxy_h, py1 + strip) - max(0, py1 - strip)))
        border_penalty += (cut_band / cut_area) * 0.9
    if cand.y2 < 0.995:
        cut_band = _rect_sum(integral, px1, py2 - strip, px2, py2 + strip)
        cut_area = max(1, max(1, px2 - px1) * (min(proxy_h, py2 + strip) - max(0, py2 - strip)))
        border_penalty += (cut_band / cut_area) * 0.9

    # Also penalize clipping primary subject box if no faces were detected
    if not analysis.faces and analysis.subject_box is not None:
        subj_inside = analysis.subject_box.fraction_inside(cand)
        if subj_inside < 0.98:
            border_penalty += 0.9 * (1.0 - subj_inside)

    # 4. Horizon placement (Rule of Thirds)
    horizon_score = 0.0
    h_est = analysis.horizon
    if h_est.y_norm is not None and h_est.confidence >= 0.30 and cand.h > 0:
        rel_h = (h_est.y_norm - cand.y) / cand.h
        if 0.12 <= rel_h <= 0.88:
            if h_est.sky_interest > h_est.ground_interest * 1.15:
                # Dramatic sky -> favor lower third (2/3 from top)
                dist_pref = abs(rel_h - 0.667)
                dist_alt = abs(rel_h - 0.333)
                third_score = max(1.0 - dist_pref * 3.0, 0.75 * (1.0 - dist_alt * 3.0), 0.0)
            elif h_est.ground_interest > h_est.sky_interest * 1.15:
                # Rich foreground -> favor upper third (1/3 from top)
                dist_pref = abs(rel_h - 0.333)
                dist_alt = abs(rel_h - 0.667)
                third_score = max(1.0 - dist_pref * 3.0, 0.75 * (1.0 - dist_alt * 3.0), 0.0)
            else:
                dist = min(abs(rel_h - 0.333), abs(rel_h - 0.667))
                third_score = max(0.0, 1.0 - dist * 3.0)
            # Gentle penalty if horizon is dead-center
            center_pen = max(0.0, 0.15 - abs(rel_h - 0.5)) * 0.8
            horizon_score = h_est.confidence * (0.55 * third_score - center_pen)
        else:
            # Horizon falls awkwardly right on or outside the crop edge
            horizon_score = -0.35 * h_est.confidence

    # 5. Composition & Rule-of-Thirds alignment
    composition_score = 0.0
    if analysis.faces:
        # Eyes/faces look most natural near the upper-third line (~0.35)
        avg_face_cy = sum(f.cy for f in analysis.faces) / len(analysis.faces)
        avg_face_cx = sum(f.cx for f in analysis.faces) / len(analysis.faces)
        rel_fy = (avg_face_cy - cand.y) / max(cand.h, 1e-6)
        rel_fx = (avg_face_cx - cand.x) / max(cand.w, 1e-6)
        eye_line_score = max(0.0, 1.0 - abs(rel_fy - 0.35) * 2.2)
        fx_dist = min(abs(rel_fx - 0.5), abs(rel_fx - 0.333), abs(rel_fx - 0.667))
        horiz_balance = max(0.0, 1.0 - fx_dist * 2.5)
        composition_score = 0.35 * eye_line_score + 0.15 * horiz_balance
    else:
        # Subject center-of-mass proximity to rule-of-thirds power lines
        scx, scy = analysis.primary_center
        rel_sx = (scx - cand.x) / max(cand.w, 1e-6)
        rel_sy = (scy - cand.y) / max(cand.h, 1e-6)
        if 0.0 <= rel_sx <= 1.0 and 0.0 <= rel_sy <= 1.0:
            dx_third = min(abs(rel_sx - 0.333), abs(rel_sx - 0.667), abs(rel_sx - 0.5) * 1.15)
            dy_third = min(abs(rel_sy - 0.333), abs(rel_sy - 0.667), abs(rel_sy - 0.5) * 1.15)
            x_align = max(0.0, 1.0 - dx_third * 2.5)
            y_align = max(0.0, 1.0 - dy_third * 2.5)
            composition_score = 0.22 * (x_align + y_align) * 0.5

    # 6. Mild center bias tie-breaker + slight preference for larger crop area
    center_dist = float(np.hypot(cand.cx - 0.5, cand.cy - 0.5))
    center_bias = max(0.0, 1.0 - center_dist * 2.0) * 0.06
    area_bonus = cand.area * 0.12

    return (
        1.35 * saliency_score
        + horizon_score
        + composition_score
        + center_bias
        + area_bonus
        - face_penalty
        - border_penalty
    )


def compute_smart_crop(rgb_image: np.ndarray, config: CropConfig) -> CropDecision:
    """Analyze a scene and compute the optimal aspect-ratio crop rectangle."""
    src_h, src_w = rgb_image.shape[:2]
    src_ratio = src_w / src_h
    target_ratio = config.aspect_ratio.value

    rel_diff = abs(src_ratio - target_ratio) / target_ratio
    if rel_diff <= config.aspect_tolerance:
        return CropDecision(
            box=CropBox(0, 0, src_w, src_h),
            was_cropped=False,
            score=1.0,
            reason=f"Already matches {config.aspect_ratio.display_name} (diff {rel_diff:.1%})",
            analysis=None,
        )

    analysis = analyze_scene(rgb_image, max_dim=config.analysis_max_dim)
    integral = cv2.integral(analysis.saliency_map.astype(np.float64))
    total_saliency = float(analysis.saliency_map.sum())

    max_w, max_h = _maximal_crop_dimensions(src_w, src_h, target_ratio)
    scales = [1.0]
    if config.allow_smart_zoom and config.min_crop_fraction < 0.99:
        scales.append(max(0.5, config.min_crop_fraction))

    candidates: list[CropBox] = []
    steps = max(5, config.candidate_steps)

    for scale in scales:
        cw = max(1, min(src_w, int(round(max_w * scale))))
        ch = max(1, min(src_h, int(round(max_h * scale))))
        max_x = src_w - cw
        max_y = src_h - ch

        if max_x > 0 and max_y == 0:
            xs = sorted({int(round(max_x * i / (steps - 1))) for i in range(steps)})
            for x in xs:
                candidates.append(CropBox(x, 0, cw, ch))
        elif max_y > 0 and max_x == 0:
            ys = sorted({int(round(max_y * i / (steps - 1))) for i in range(steps)})
            for y in ys:
                candidates.append(CropBox(0, y, cw, ch))
        elif max_x > 0 and max_y > 0:
            grid = max(5, int(round(np.sqrt(steps))))
            xs = sorted({int(round(max_x * i / (grid - 1))) for i in range(grid)})
            ys = sorted({int(round(max_y * j / (grid - 1))) for j in range(grid)})
            for x in xs:
                for y in ys:
                    candidates.append(CropBox(x, y, cw, ch))
        else:
            candidates.append(CropBox(0, 0, cw, ch))

    best_box = candidates[0]
    best_score = float("-inf")
    for box in candidates:
        norm_box = Box(
            x=box.x / src_w,
            y=box.y / src_h,
            w=box.width / src_w,
            h=box.height / src_h,
        )
        score = _score_candidate(norm_box, analysis, integral, total_saliency, src_w, src_h)
        if score > best_score:
            best_score = score
            best_box = box

    # Build a concise human-readable explanation
    notes: list[str] = []
    if analysis.faces:
        notes.append(f"{len(analysis.faces)} face(s) protected")
    if analysis.horizon.y_norm is not None and analysis.horizon.confidence >= 0.30:
        rel_h = (analysis.horizon.y_norm * src_h - best_box.y) / max(1, best_box.height)
        notes.append(f"horizon @ {rel_h:.0%} (conf {analysis.horizon.confidence:.2f})")
    if not notes:
        notes.append(
            f"salient focus ({analysis.primary_center[0]:.0%}, {analysis.primary_center[1]:.0%})"
        )
    notes.append(
        f"cropped {src_w}x{src_h} -> {best_box.width}x{best_box.height} "
        f"@ ({best_box.x},{best_box.y})"
    )

    return CropDecision(
        box=best_box,
        was_cropped=True,
        score=best_score,
        reason="; ".join(notes),
        analysis=analysis,
    )


def crop_pil_image(image: Image.Image, config: CropConfig) -> tuple[Image.Image, CropDecision]:
    """Run smart scene analysis and return the cropped PIL image and decision metadata."""
    rgb_np = np.asarray(image.convert("RGB"), dtype=np.uint8)
    decision = compute_smart_crop(rgb_np, config)
    if not decision.was_cropped:
        return image, decision
    return image.crop(decision.box.as_pil_box), decision
