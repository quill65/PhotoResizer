"""Tests for mismatched orientation modes: match_orientation, blur_fill, and ai_outpaint."""

import base64
import io
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import pytest
from PIL import Image

from photo_resizer.config import AspectRatio, CropConfig, EnhanceConfig, ProcessConfig
from photo_resizer.enhancer import ModelUnavailableError, post_gemini_image_request
from photo_resizer.outpainter import (
    GeminiOutpainter,
    align_and_composite_original,
    compute_expanded_canvas_size,
    create_blurred_backdrop,
    is_orientation_mismatch,
    nearest_gemini_aspect_ratio,
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


def test_ai_outpaint_preserves_seamless_output_and_aligned_original(tmp_path: Path) -> None:
    """Verify SIFT alignment composites cleanly when matched and avoids seams when reframed."""
    assert nearest_gemini_aspect_ratio(16 / 9) == "16:9"
    assert nearest_gemini_aspect_ratio(4 / 3) == "4:3"
    assert nearest_gemini_aspect_ratio(9 / 16) == "9:16"

    # Build a textured synthetic portrait (rich features for SIFT) inside a 16:9 canvas
    rng = np.random.default_rng(123)
    full_scene = np.full((600, 1066, 3), (50, 120, 60), dtype=np.uint8)
    for _ in range(80):
        cx = int(rng.integers(50, 1016))
        cy = int(rng.integers(50, 550))
        rad = int(rng.integers(12, 45))
        color = tuple(int(c) for c in rng.integers(30, 240, size=3))
        import cv2

        cv2.circle(full_scene, (cx, cy), rad, color, -1)

    # Original portrait is the center 450x600 slice of full_scene
    orig_arr = full_scene[:, 308:758].copy()
    portrait = Image.fromarray(orig_arr, mode="RGB")
    src_file = tmp_path / "subject_portrait.png"
    portrait.save(src_file)

    # 1. Aligned Gemini output (resized to 1280x720) -> SIFT locks & composites seamlessly!
    aligned_ai = Image.fromarray(full_scene, mode="RGB").resize(
        (1280, 720), Image.Resampling.LANCZOS
    )
    buf = io.BytesIO()
    aligned_ai.save(buf, format="PNG")
    fake_png = buf.getvalue()

    cfg = ProcessConfig(
        source_dir=tmp_path,
        crop=CropConfig(aspect_ratio=AspectRatio(16, 9), orientation_mode="ai_outpaint"),
        enhance=EnhanceConfig(enabled=False, gemini_api_key="test-fake-key"),
    )

    with patch.object(GeminiOutpainter, "_call_gemini_rest", return_value=fake_png) as mock_api:
        summary = process_folder(cfg)

    assert mock_api.call_count == 1
    assert mock_api.call_args.kwargs.get("aspect_ratio") == "16:9"
    assert summary.processed == 1
    res = summary.results[0]
    assert res.final_size == (1067, 600)
    assert "ai_outpaint_background" in res.change_types
    assert any("Aligned Original Composite" in t for t in res.tools_used)

    # 2. Reframed/shifted Gemini output that doesn't match -> must NOT blindly stamp a seam!
    unaligned_ai = Image.new("RGB", (1280, 720), (60, 150, 230))
    seamless_out, did_comp = align_and_composite_original(unaligned_ai, portrait, 1067, 600)
    assert did_comp is False
    assert seamless_out.size == (1067, 600)
    assert np.allclose(np.asarray(seamless_out)[300, 533], [60, 150, 230], atol=2)


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


def test_gemini_429_zero_quota_fails_fast_with_clear_billing_message() -> None:
    zero_quota_resp = httpx.Response(
        429,
        json={
            "error": {
                "code": 429,
                "message": (
                    "Quota exceeded for metric: generate_content_free_tier_requests, limit: 0"
                ),
            }
        },
    )
    with (
        patch("photo_resizer.enhancer.httpx.post", return_value=zero_quota_resp) as mock_post,
        patch("photo_resizer.enhancer.time.sleep") as mock_sleep,
        pytest.raises(ModelUnavailableError, match="quota is 0"),
    ):
        post_gemini_image_request(
            "fake-key", "gemini-2.5-flash-image", b"png", "outpaint", min_interval_sec=0.0
        )

    assert mock_post.call_count == 1
    assert mock_sleep.call_count == 0


def test_gemini_429_transient_rate_limit_retries_and_succeeds() -> None:
    rate_limit_resp = httpx.Response(
        429,
        json={
            "error": {
                "code": 429,
                "message": "Resource exhausted. Please retry in 1s.",
                "details": [{"retryDelay": "1s"}],
            }
        },
    )
    ok_resp = httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": base64.b64encode(b"fake-out-image").decode("ascii"),
                                }
                            }
                        ]
                    }
                }
            ]
        },
    )
    with (
        patch("photo_resizer.enhancer.httpx.post", side_effect=[rate_limit_resp, ok_resp]) as mp,
        patch("photo_resizer.enhancer.time.sleep") as mock_sleep,
    ):
        out = post_gemini_image_request(
            "fake-key", "gemini-2.5-flash-image", b"png", "outpaint", min_interval_sec=0.0
        )

    assert out == b"fake-out-image"
    assert mp.call_count == 2
    assert mock_sleep.call_count == 1


def test_gemini_404_falls_back_from_nano_banana_alias_to_31_flash_image() -> None:
    not_found = httpx.Response(404, json={"error": {"code": 404, "message": "Model not found"}})
    ok_resp = httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": base64.b64encode(b"ok-bytes").decode("ascii"),
                                }
                            }
                        ]
                    }
                }
            ]
        },
    )
    with patch("photo_resizer.enhancer.httpx.post", side_effect=[not_found, ok_resp]) as mp:
        out = post_gemini_image_request(
            "fake-key", "gemini-nano-banana-2.1", b"png", "outpaint", min_interval_sec=0.0
        )

    assert out == b"ok-bytes"
    assert mp.call_count == 2
    assert "gemini-3.1-flash-image" in mp.call_args_list[1].args[0]


