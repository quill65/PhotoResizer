"""Tests for tiered upsampling and enhancement backends."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from photo_resizer.config import EnhanceConfig
from photo_resizer.enhancer import (
    OnnxSuperResEnhancer,
    OpenCVEnhancer,
    TieredEnhancer,
    compute_upscale_target,
)


def test_compute_upscale_target_thresholds() -> None:
    cfg = EnhanceConfig(min_long_edge=1920, min_short_edge=1080, max_upscale_factor=4.0)

    # Already sufficient -> no upsampling
    assert compute_upscale_target(2400, 1350, cfg) == (2400, 1350, 1.0)

    # Low-res 960x540 -> exact 2.0x upscale to 1920x1080
    tw, th, scale = compute_upscale_target(960, 540, cfg)
    assert (tw, th) == (1920, 1080)
    assert scale == pytest.approx(2.0)

    # Tiny 200x100 -> capped at max_upscale_factor (4.0x)
    tw2, th2, scale2 = compute_upscale_target(200, 100, cfg)
    assert (tw2, th2) == (800, 400)
    assert scale2 == pytest.approx(4.0)


def test_opencv_enhancer_rgb_and_rgba() -> None:
    enhancer = OpenCVEnhancer()
    rgb = Image.new("RGB", (320, 180), color=(90, 140, 200))
    out_rgb = enhancer.enhance(rgb, 640, 360)
    assert out_rgb.size == (640, 360)
    assert out_rgb.mode == "RGB"

    rgba = Image.new("RGBA", (200, 200), color=(50, 100, 150, 180))
    out_rgba = enhancer.enhance(rgba, 400, 400)
    assert out_rgba.size == (400, 400)
    assert out_rgba.mode == "RGBA"


def test_onnx_enhancer_tiled_blending_with_mock_session() -> None:
    """Verify tiled inference, overlap feathering, and exact resize using a mock 2x ORT session."""

    class FakeIO:
        def __init__(self, name: str, shape: list[object]) -> None:
            self.name = name
            self.shape = shape

    class FakeOrtSession:
        def get_inputs(self) -> list[FakeIO]:
            return [FakeIO("input", [1, 3, "h", "w"])]

        def get_outputs(self) -> list[FakeIO]:
            return [FakeIO("output", [1, 3, "oh", "ow"])]

        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

        def run(self, _outputs: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            x = feed["input"]  # (1, 3, h, w)
            # Simple deterministic 2x nearest upsample in NCHW
            up = np.repeat(np.repeat(x, 2, axis=2), 2, axis=3)
            return [up.astype(np.float32)]

    enhancer = OnnxSuperResEnhancer(tile_size=64, tile_overlap=16, auto_download=False)
    fake_sess = FakeOrtSession()
    enhancer._session = fake_sess  # type: ignore[assignment]
    enhancer._input_name = "input"
    enhancer._output_name = "output"
    enhancer._model_scale = enhancer._probe_scale(fake_sess)  # type: ignore[arg-type]
    assert enhancer._model_scale == 2

    # Image larger than tile_size (110x90) exercises multi-tile overlap blending
    arr = np.full((90, 110, 3), (80, 160, 220), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    out = enhancer.enhance(img, 200, 164)
    assert out.size == (200, 164)
    out_arr = np.asarray(out)
    # Uniform color should remain uniform with zero seam artifacts
    assert np.allclose(out_arr.mean(axis=(0, 1)), [80, 160, 220], atol=2.0)


def test_tiered_enhancer_routing_and_offline_fallback(tmp_path: Path) -> None:
    # Mild upscale (< 1.4x) uses OpenCV directly
    cfg_mild = EnhanceConfig(
        mode="auto",
        min_long_edge=1200,
        min_short_edge=675,
        mild_upscale_threshold=1.4,
        auto_download_onnx=False,
    )
    mild_img = Image.new("RGB", (1000, 562), (40, 80, 120))
    res_mild = TieredEnhancer(cfg_mild).enhance_if_needed(mild_img, target_ratio=16 / 9)
    assert res_mild.was_enhanced is True
    assert res_mild.backend == "opencv"
    assert res_mild.image.size == (1200, 675)

    # Larger upscale (>= 1.4x) with missing ONNX falls back cleanly to OpenCV
    cfg_fallback = EnhanceConfig(
        mode="offline",
        min_long_edge=1600,
        min_short_edge=900,
        onnx_model_path=tmp_path / "missing.onnx",
        auto_download_onnx=False,
    )
    fb_img = Image.new("RGB", (800, 450), (40, 80, 120))
    res_fb = TieredEnhancer(cfg_fallback).enhance_if_needed(fb_img, target_ratio=16 / 9)
    assert res_fb.was_enhanced is True
    assert res_fb.backend == "opencv"
    assert "onnx skipped" in res_fb.note
    assert res_fb.image.size == (1600, 900)

    # Gemini mode without API key falls back cleanly to offline OpenCV
    cfg_gemini_no_key = EnhanceConfig(
        mode="gemini",
        min_long_edge=1200,
        min_short_edge=675,
        gemini_api_key=None,
        onnx_model_path=tmp_path / "missing.onnx",
        auto_download_onnx=False,
    )
    res_gem = TieredEnhancer(cfg_gemini_no_key).enhance_if_needed(
        Image.new("RGB", (600, 338), (100, 100, 100))
    )
    assert res_gem.was_enhanced is True
    assert res_gem.backend == "opencv"
    assert "gemini skipped" in res_gem.note
