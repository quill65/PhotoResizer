"""Smart aspect-ratio cropper driven by offline scene analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from photo_resizer.analyzer import Box, SceneAnalysis, analyze_scene
from photo_resizer.config import CropConfig, FramingStyle


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
    was_smart_zoomed: bool = False
    horizon_y_ratio: float | None = None
    subject_clipped_fraction: float = 0.0
    would_clip_subject: bool = False


def load_oriented_image_with_meta(path: Path) -> tuple[Image.Image, bytes | None, bool]:
    """Load an image with EXIF auto-orientation applied, returning (image, exif, was_rotated)."""
    was_transposed = False
    with Image.open(path) as img:
        raw_exif = img.info.get("exif")
        if raw_exif:
            try:
                exif_check = Image.Exif()
                exif_check.load(raw_exif)
                orientation = exif_check.get(274, 1)
                was_transposed = isinstance(orientation, int) and orientation not in (0, 1)
            except Exception:
                was_transposed = False

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
    return oriented, cleaned_exif, was_transposed


def load_oriented_image(path: Path) -> tuple[Image.Image, bytes | None]:
    """Load an image with EXIF auto-orientation applied and safe EXIF preserved."""
    oriented, cleaned_exif, _ = load_oriented_image_with_meta(path)
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
    framing_style: FramingStyle = "center_subject",
) -> float:
    """Score a candidate normalized crop box for visual composition quality."""
    del img_w, img_h
    proxy_w, proxy_h = analysis.proxy_size
    px1 = int(round(cand.x * proxy_w))
    py1 = int(round(cand.y * proxy_h))
    px2 = int(round(cand.x2 * proxy_w))
    py2 = int(round(cand.y2 * proxy_h))

    # 1. Saliency capture fraction (0..1)
    captured = _rect_sum(integral, px1, py1, px2, py2)
    saliency_score = captured / max(total_saliency, 1e-6)

    # 2. Animal, face, & headroom protection (strictly avoid clipping any part of a subject)
    subject_clip_penalty = 0.0
    for animal in analysis.animal_boxes:
        inside = animal.fraction_inside(cand)
        if inside < 0.999:
            subject_clip_penalty += 6.5 * (1.0 - inside)
    for margin in analysis.animal_margin_boxes:
        inside = margin.fraction_inside(cand)
        if inside < 0.999:
            subject_clip_penalty += 2.0 * (1.0 - inside)

    for face in analysis.faces:
        inside = face.fraction_inside(cand)
        if inside < 0.999:
            subject_clip_penalty += 4.5 * (1.0 - inside)
    for headroom in analysis.headroom_boxes:
        inside = headroom.fraction_inside(cand)
        if inside < 0.999:
            subject_clip_penalty += 1.6 * (1.0 - inside)

    # 3. Subject edge-slicing penalty: penalize cuts passing through dense saliency
    border_penalty = 0.0
    strip = max(2, int(round(min(proxy_w, proxy_h) * 0.025)))
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

    # Protect primary foreground subject box (covers animals, flowers, tails, petals)
    if analysis.subject_box is not None:
        subj_inside = analysis.subject_box.fraction_inside(cand)
        if subj_inside < 0.995:
            weight = 4.2 if analysis.has_strong_subject else 1.4
            subject_clip_penalty += weight * (1.0 - subj_inside)

    # 4. Horizon placement (Rule of Thirds, mainly for landscapes without a dominant close subject)
    horizon_score = 0.0
    h_est = analysis.horizon
    if h_est.y_norm is not None and h_est.confidence >= 0.30 and cand.h > 0:
        rel_h = (h_est.y_norm - cand.y) / cand.h
        if 0.12 <= rel_h <= 0.88:
            if h_est.sky_interest > h_est.ground_interest * 1.15:
                dist_pref = abs(rel_h - 0.667)
                dist_alt = abs(rel_h - 0.333)
                third_score = max(1.0 - dist_pref * 3.0, 0.75 * (1.0 - dist_alt * 3.0), 0.0)
            elif h_est.ground_interest > h_est.sky_interest * 1.15:
                dist_pref = abs(rel_h - 0.333)
                dist_alt = abs(rel_h - 0.667)
                third_score = max(1.0 - dist_pref * 3.0, 0.75 * (1.0 - dist_alt * 3.0), 0.0)
            else:
                dist = min(abs(rel_h - 0.333), abs(rel_h - 0.667))
                third_score = max(0.0, 1.0 - dist * 3.0)
            center_pen = max(0.0, 0.15 - abs(rel_h - 0.5)) * 0.8
            horizon_score = h_est.confidence * (0.55 * third_score - center_pen)
        else:
            horizon_score = -0.35 * h_est.confidence

    # 5. Subject Framing: strongly center animals/subjects or align on rule of thirds
    composition_score = 0.0
    scx, scy = analysis.primary_center
    rel_sx = (scx - cand.x) / max(cand.w, 1e-6)
    rel_sy = (scy - cand.y) / max(cand.h, 1e-6)
    subject_in_frame = 0.0 <= rel_sx <= 1.0 and 0.0 <= rel_sy <= 1.0

    if analysis.faces and not analysis.animal_boxes:
        avg_face_cy = sum(f.cy for f in analysis.faces) / len(analysis.faces)
        avg_face_cx = sum(f.cx for f in analysis.faces) / len(analysis.faces)
        rel_fy = (avg_face_cy - cand.y) / max(cand.h, 1e-6)
        rel_fx = (avg_face_cx - cand.x) / max(cand.w, 1e-6)
        if framing_style == "center_subject":
            eye_line_score = max(0.0, 1.0 - abs(rel_fy - 0.42) * 2.0)
            horiz_balance = max(0.0, 1.0 - abs(rel_fx - 0.5) * 2.2)
            composition_score = 0.90 * horiz_balance + 0.55 * eye_line_score
        else:
            eye_line_score = max(0.0, 1.0 - abs(rel_fy - 0.35) * 2.2)
            fx_dist = min(abs(rel_fx - 0.5), abs(rel_fx - 0.333), abs(rel_fx - 0.667))
            horiz_balance = max(0.0, 1.0 - fx_dist * 2.5)
            composition_score = 0.40 * eye_line_score + 0.25 * horiz_balance
    elif subject_in_frame:
        if framing_style == "center_subject":
            # Actively center animals / primary subjects inside the crop window
            dist_center = float(np.hypot(rel_sx - 0.5, rel_sy - 0.5))
            center_align = max(0.0, 1.0 - dist_center * 1.9)
            # Strong weight when a neural/animal subject is detected so background foliage
            # never pulls the crop away from the bird/insect/mammal.
            weight = 2.40 if analysis.has_strong_subject else 0.95
            composition_score = weight * center_align
        else:
            dx_third = min(abs(rel_sx - 0.333), abs(rel_sx - 0.667), abs(rel_sx - 0.5) * 1.15)
            dy_third = min(abs(rel_sy - 0.333), abs(rel_sy - 0.667), abs(rel_sy - 0.5) * 1.15)
            x_align = max(0.0, 1.0 - dx_third * 2.5)
            y_align = max(0.0, 1.0 - dy_third * 2.5)
            weight = 1.25 if analysis.has_strong_subject else 0.35
            composition_score = weight * (x_align + y_align) * 0.5
    elif analysis.has_strong_subject:
        # Strongly penalize crops that leave the detected subject center completely outside
        composition_score = -2.5

    # 6. Mild frame center tie-breaker + slight preference for larger crop area
    center_dist = float(np.hypot(cand.cx - 0.5, cand.cy - 0.5))
    center_bias = max(0.0, 1.0 - center_dist * 2.0) * 0.05
    area_bonus = cand.area * 0.12

    # When a clear animal/subject was detected, let subject protection + framing lead
    # rather than summing noisy background foliage across the whole image.
    saliency_weight = 0.85 if analysis.has_strong_subject else 1.35

    return (
        saliency_weight * saliency_score
        + horizon_score
        + composition_score
        + center_bias
        + area_bonus
        - subject_clip_penalty
        - border_penalty
    )


def _measure_subject_clipping(
    best_box: CropBox,
    src_w: int,
    src_h: int,
    analysis: SceneAnalysis,
) -> float:
    """Compute fraction (0.0..1.0) of detected subject(s) clipped outside best_box."""
    norm_crop = Box(
        x=best_box.x / src_w,
        y=best_box.y / src_h,
        w=best_box.width / src_w,
        h=best_box.height / src_h,
    )

    worst_clipped = 0.0
    protected: list[Box] = list(analysis.animal_boxes) + list(analysis.faces)
    if analysis.has_strong_subject and analysis.subject_box is not None:
        protected.append(analysis.subject_box)

    for subj in protected:
        clipped = max(0.0, 1.0 - subj.fraction_inside(norm_crop))
        worst_clipped = max(worst_clipped, clipped)

    # Also measure foreground mask mass cut outside the crop (catches flower petals,
    # bird wings/tails, or insect antennae directly on the neural mask).
    fg_mask = analysis.foreground_mask
    if fg_mask is not None:
        fh, fw = fg_mask.shape
        fg_bin = np.where(fg_mask >= 0.25, fg_mask, 0.0)
        total_fg = float(fg_bin.sum())
        if total_fg > 1e-6:
            x1 = max(0, min(fw, int(round(norm_crop.x * fw))))
            y1 = max(0, min(fh, int(round(norm_crop.y * fh))))
            x2 = max(x1, min(fw, int(round(norm_crop.x2 * fw))))
            y2 = max(y1, min(fh, int(round(norm_crop.y2 * fh))))
            inside_fg = float(fg_bin[y1:y2, x1:x2].sum())
            mask_clipped = max(0.0, 1.0 - (inside_fg / total_fg))
            worst_clipped = max(worst_clipped, mask_clipped)

    return worst_clipped


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

    analysis = analyze_scene(
        rgb_image,
        max_dim=config.analysis_max_dim,
        use_neural_detector=config.use_neural_detector,
        auto_download_detectors=config.auto_download_detectors,
    )
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
        score = _score_candidate(
            norm_box,
            analysis,
            integral,
            total_saliency,
            src_w,
            src_h,
            framing_style=config.framing_style,
        )
        if score > best_score:
            best_score = score
            best_box = box

    # Build a concise human-readable explanation
    notes: list[str] = []
    horizon_rel: float | None = None
    if analysis.animal_boxes:
        labels = sorted({b.label for b in analysis.animal_boxes})
        top_conf = max(b.score for b in analysis.animal_boxes)
        notes.append(f"detected {'/'.join(labels)} ({top_conf:.2f})")
    elif analysis.has_strong_subject and "u2netp_foreground" in analysis.detector_tags:
        notes.append(
            f"foreground subject @ ({analysis.primary_center[0]:.0%}, "
            f"{analysis.primary_center[1]:.0%})"
        )
    if analysis.faces:
        notes.append(f"{len(analysis.faces)} face(s) protected")
    if analysis.horizon.y_norm is not None and analysis.horizon.confidence >= 0.30:
        horizon_rel = (analysis.horizon.y_norm * src_h - best_box.y) / max(1, best_box.height)
        notes.append(f"horizon @ {horizon_rel:.0%} (conf {analysis.horizon.confidence:.2f})")
    if not notes:
        notes.append(
            f"salient focus ({analysis.primary_center[0]:.0%}, {analysis.primary_center[1]:.0%})"
        )
    was_smart_zoomed = best_box.width < max_w - 1 or best_box.height < max_h - 1
    if was_smart_zoomed:
        notes.append("smart zoom reframed")
    notes.append(
        f"cropped {src_w}x{src_h} -> {best_box.width}x{best_box.height} "
        f"@ ({best_box.x},{best_box.y})"
    )

    clipped_frac = _measure_subject_clipping(best_box, src_w, src_h, analysis)
    would_clip = clipped_frac > max(0.0, config.max_subject_clip_fraction)

    return CropDecision(
        box=best_box,
        was_cropped=True,
        score=best_score,
        reason="; ".join(notes),
        analysis=analysis,
        was_smart_zoomed=was_smart_zoomed,
        horizon_y_ratio=horizon_rel,
        subject_clipped_fraction=clipped_frac,
        would_clip_subject=would_clip,
    )


def crop_pil_image(image: Image.Image, config: CropConfig) -> tuple[Image.Image, CropDecision]:
    """Run smart scene analysis and return the cropped PIL image and decision metadata."""
    rgb_np = np.asarray(image.convert("RGB"), dtype=np.uint8)
    decision = compute_smart_crop(rgb_np, config)
    if not decision.was_cropped:
        return image, decision
    return image.crop(decision.box.as_pil_box), decision
