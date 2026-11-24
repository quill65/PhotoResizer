"""Tests for offline scene analysis, smart cropping, and EXIF handling."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from photo_resizer.analyzer import Box, HorizonEstimate, SceneAnalysis, analyze_scene
from photo_resizer.config import AspectRatio, CropConfig
from photo_resizer.cropper import (
    _score_candidate,
    compute_smart_crop,
    load_oriented_image,
)


def test_already_matching_aspect_ratio_skips_crop() -> None:
    img = np.full((900, 1600, 3), 120, dtype=np.uint8)
    decision = compute_smart_crop(img, CropConfig(aspect_ratio=AspectRatio(16, 9)))
    assert decision.was_cropped is False
    assert (decision.box.width, decision.box.height) == (1600, 900)


def test_off_center_subject_is_preserved_in_crop() -> None:
    """A bright textured subject near the right edge must not be clipped when cropping to 1:1."""
    h, w = 600, 1200
    img = np.full((h, w, 3), (35, 45, 55), dtype=np.uint8)

    # Draw a high-contrast detailed subject on the far right (x in [880, 1120])
    cv2.circle(img, (1000, 300), 110, (240, 180, 30), -1)
    cv2.circle(img, (1000, 300), 70, (220, 40, 40), 12)
    cv2.rectangle(img, (945, 245), (1055, 355), (255, 255, 255), 6)

    decision = compute_smart_crop(
        img, CropConfig(aspect_ratio=AspectRatio(1, 1), use_neural_detector=False)
    )
    assert decision.was_cropped is True
    assert decision.box.width == 600
    assert decision.box.height == 600
    # Crop must shift right so the subject center (x=1000) is comfortably inside
    assert decision.box.x >= 500
    assert decision.box.x <= 880
    assert decision.box.x + decision.box.width >= 1090


def test_landscape_horizon_detection_and_thirds_framing() -> None:
    """A landscape with a clear horizon should detect it and avoid clipping it to an edge."""
    h, w = 900, 900
    img = np.zeros((h, w, 3), dtype=np.uint8)
    horizon_y = 360  # 40% from top

    # Smooth blue sky above horizon
    img[:horizon_y, :] = (135, 190, 240)
    # Rich textured green/brown ground below horizon
    rng = np.random.default_rng(42)
    ground = np.full((h - horizon_y, w, 3), (45, 115, 55), dtype=np.uint8)
    noise = rng.integers(-35, 36, size=ground.shape, dtype=np.int16)
    img[horizon_y:, :] = np.clip(ground.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    analysis = analyze_scene(img, max_dim=450, use_neural_detector=False)
    assert analysis.horizon.y_norm is not None
    assert analysis.horizon.y_norm == pytest.approx(horizon_y / h, abs=0.06)

    # Crop square landscape into 16:9 widescreen -> vertical crop height ~506px
    decision = compute_smart_crop(
        img, CropConfig(aspect_ratio=AspectRatio(16, 9), use_neural_detector=False)
    )
    assert decision.was_cropped is True
    rel_horizon = (horizon_y - decision.box.y) / decision.box.height
    # Horizon should stay well inside the frame, near upper third (~0.25..0.55)
    assert 0.20 <= rel_horizon <= 0.60


def test_face_and_headroom_protection_strongly_penalizes_clipping() -> None:
    """Verify scorer strongly prefers keeping a detected face + headroom intact over clipping."""
    saliency = np.full((100, 100), 0.2, dtype=np.float32)
    face = Box(x=0.35, y=0.10, w=0.25, h=0.25, label="face")
    headroom = Box(x=0.26, y=0.0, w=0.43, h=0.55, label="headroom")
    analysis = SceneAnalysis(
        faces=[face],
        headroom_boxes=[headroom],
        saliency_map=saliency,
        horizon=HorizonEstimate(None, 0.0),
        proxy_size=(100, 100),
    )
    integral = cv2.integral(saliency.astype(np.float64))
    total = float(saliency.sum())

    top_crop = Box(x=0.0, y=0.0, w=1.0, h=0.56)
    clipped_face_crop = Box(x=0.0, y=0.22, w=1.0, h=0.56)

    score_keep = _score_candidate(top_crop, analysis, integral, total, 1000, 1000)
    score_clip = _score_candidate(clipped_face_crop, analysis, integral, total, 1000, 1000)
    assert score_keep > score_clip + 1.0


def test_load_oriented_image_applies_exif_rotation_and_resets_tag(tmp_path: Path) -> None:
    img_path = tmp_path / "rotated.jpg"
    raw = Image.new("RGB", (400, 200), color=(20, 120, 200))
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 CW
    raw.save(img_path, format="JPEG", exif=exif.tobytes())

    oriented, cleaned_exif = load_oriented_image(img_path)
    # After 90-degree transpose, 400x200 becomes 200x400
    assert oriented.size == (200, 400)
    assert cleaned_exif is not None
    parsed = Image.Exif()
    parsed.load(cleaned_exif)
    assert parsed.get(274) == 1
