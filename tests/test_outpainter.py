"""Tests for mismatched orientation modes: match_orientation, blur_fill, and ai_outpaint."""

import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from photo_resizer.config import AspectRatio, CropConfig, EnhanceConfig, ProcessConfig
from photo_resizer.outpainter import (
    GeminiOutpainter,
    compute_expanded_canvas_size,
    create_blurred_backdrop,
    is_orientation_mismatch,
)
from photo_resizer.processor import process_folder


def test_is_orientation_mismatch_cases() -> None:
    # Portrait image vs landscape target -> mismatch
    assert is_orientation_mismatch(1000, 1500, 16 / 9) is True
    # Landscape image vs portrait target -> mismatch
    assert is_orientation_mismatch(1500, 1000, 9 / 16) is True
    # Matching orientations -> no mismatch
    assert is_orientation_mismatch(1600, 1200, 16 / 9) is False
    assert is_orientation_mismatch(1200, 1600, 9 / 16) is False
    # Square target or square image -> no portrait/landscape mismatch
    assert is_orientation_mismatch(1000, 1500, 1.0) is False
    assert is_orientation_mismatch(1000, 1000, 16 / 9) is False


def test_compute_expanded_canvas_size() -> None:
    # Portrait 600x800 expanded to 16:9 -> 1422x800 centered
    cw, ch, ox, oy = compute_expanded_canvas_size(600, 800, 16 / 9)
    assert (cw, ch) == (1422, 800)
    assert ox == (1422 - 600) // 2
    assert oy == 0


def test_blur_fill_keeps_center_portrait_sharp_and_fills_landscape(tmp_path: Path) -> None:
    # Distinct solid red portrait photo
    portrait = Image.new("RGB", (600, 800), (210, 40, 40))
    expanded = create_blurred_backdrop(portrait, 16 / 9)
    assert expanded.size == (1422, 800)

    arr = np.asarray(expanded)
    # Center of image preserves untouched original portrait pixels
    center_rgb = arr[400, 711]
    assert np.allclose(center_rgb, [210, 40, 40], atol=2)
    # Left side backdrop is darker blurred version of photo colors
    side_rgb = arr[400, 30]
    assert side_rgb[0] > side_rgb[1] + 50
    assert side_rgb[0] < 180

    # End-to-end via process_folder
    src_file = tmp_path / "portrait.png"
    portrait.save(src_file)
    cfg = ProcessConfig(
        source_dir=tmp_path,
        crop=CropConfig(aspect_ratio=AspectRatio(16, 9), orientation_mode="blur_fill"),
        enhance=EnhanceConfig(enabled=False),
    )
    summary = process_folder(cfg)
    assert summary.processed == 1
    res = summary.results[0]
    assert res.final_size == (1422, 800)
    assert "blurred_backdrop_fill" in res.change_types
    assert any("blurred_backdrop_fill" in t for t in res.tools_used)


def test_match_orientation_flips_ratio_for_portrait(tmp_path: Path) -> None:
    src_file = tmp_path / "tall.jpg"
    Image.new("RGB", (1800, 2400), (70, 120, 190)).save(src_file)

    cfg = ProcessConfig(
        source_dir=tmp_path,
        crop=CropConfig(aspect_ratio=AspectRatio(16, 9), orientation_mode="match_orientation"),
        enhance=EnhanceConfig(enabled=False),
    )
    summary = process_folder(cfg)
    assert summary.processed == 1
    res = summary.results[0]
    # 16:9 flipped to 9:16 -> 2400 * 9/16 = 1350 x 2400
    assert res.final_size == (1350, 2400)
    assert "orientation_matched" in res.change_types
    assert "aspect_ratio_crop" in res.change_types


def test_ai_outpaint_preserves_original_center_and_logs_gemini_tool(tmp_path: Path) -> None:
    src_file = tmp_path / "subject_portrait.png"
    # Portrait with unique bright yellow color so we verify untouched central compositing
    portrait = Image.new("RGB", (600, 800), (245, 220, 30))
    portrait.save(src_file)

    # Mock Gemini returning a blue-sky landscape image
    fake_ai_img = Image.new("RGB", (1280, 720), (60, 150, 230))
    buf = io.BytesIO()
    fake_ai_img.save(buf, format="PNG")
    fake_png = buf.getvalue()

    cfg = ProcessConfig(
        source_dir=tmp_path,
        crop=CropConfig(aspect_ratio=AspectRatio(16, 9), orientation_mode="ai_outpaint"),
        enhance=EnhanceConfig(enabled=False, gemini_api_key="test-fake-key"),
    )

    with patch.object(GeminiOutpainter, "_call_gemini_rest", return_value=fake_png) as mock_api:
        summary = process_folder(cfg)

    assert mock_api.call_count == 1
    assert summary.processed == 1
    res = summary.results[0]
    assert res.final_size == (1422, 800)
    assert "ai_outpaint_background" in res.change_types
    assert any("Gemini AI Outpaint" in t for t in res.tools_used)

    with Image.open(tmp_path / "corrected" / "subject_portrait.png") as out_img:
        arr = np.asarray(out_img)
        # Central original portrait pixels must remain untouched yellow
        assert np.allclose(arr[400, 711], [245, 220, 30], atol=2)
        # Outer left/right wings must contain the AI-generated blue background
        assert np.allclose(arr[400, 40], [60, 150, 230], atol=5)
        assert np.allclose(arr[400, 1380], [60, 150, 230], atol=5)


def test_ai_outpaint_without_api_key_falls_back_to_blur_fill(tmp_path: Path) -> None:
    src_file = tmp_path / "portrait_offline.png"
    Image.new("RGB", (600, 800), (90, 160, 110)).save(src_file)

    cfg = ProcessConfig(
        source_dir=tmp_path,
        crop=CropConfig(aspect_ratio=AspectRatio(16, 9), orientation_mode="ai_outpaint"),
        enhance=EnhanceConfig(enabled=False, gemini_api_key=None),
    )
    with patch.dict("os.environ", {}, clear=False):
        # Ensure GEMINI_API_KEY isn't picked up from ambient environment
        import os

        os.environ.pop("GEMINI_API_KEY", None)
        summary = process_folder(cfg)

    assert summary.processed == 1
    res = summary.results[0]
    assert res.final_size == (1422, 800)
    assert "blurred_backdrop_fill" in res.change_types
    assert "ai_outpaint skipped" in res.message
