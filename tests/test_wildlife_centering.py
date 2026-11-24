"""Tests for offline neural wildlife detection (YOLO11n + U2-Net-P) and subject centering."""

import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from photo_resizer.config import AspectRatio, CropConfig, EnhanceConfig, ProcessConfig
from photo_resizer.cropper import compute_smart_crop
from photo_resizer.outpainter import GeminiOutpainter, compute_expanded_canvas_size
from photo_resizer.processor import process_folder
from photo_resizer.subject_detector import (
    DetectedSubject,
    NeuralSubjectResult,
    _U2NetSegmenter,
    _YoloAnimalDetector,
)


def _busy_foliage_background(h: int, w: int, seed: int = 7) -> np.ndarray:
    """Create a heavily textured leafy/branchy background that fools naive edge saliency."""
    rng = np.random.default_rng(seed)
    base = np.full((h, w, 3), (45, 95, 40), dtype=np.int16)
    # Strong texture across the left half to try pulling naive croppers away from a right subject
    noise = rng.integers(-60, 61, size=(h, w, 3), dtype=np.int16)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    return img


def test_bird_detection_centers_animal_despite_busy_foliage() -> None:
    """A bird perched off-center amidst dense foliage should be cleanly centered, not clipped."""
    h, w = 800, 1200
    img = _busy_foliage_background(h, w)

    # Bird sits near x=0.72 (around pixel 864 of 1200), size ~12% x 18%
    fake_neural = NeuralSubjectResult(
        animals=[DetectedSubject(x=0.66, y=0.38, w=0.12, h=0.18, score=0.91, label="bird")],
        detector_tags=["yolo11n:bird"],
    )

    with patch("photo_resizer.analyzer.get_wildlife_detector") as mock_get:
        mock_get.return_value.detect.return_value = fake_neural
        # Crop 1200x800 -> 1:1 (800x800). Shifting to x=400 places bird (864px) near center (0.58)
        decision = compute_smart_crop(
            img,
            CropConfig(
                aspect_ratio=AspectRatio(1, 1),
                framing_style="center_subject",
                auto_download_detectors=False,
            ),
        )

    assert decision.was_cropped is True
    assert decision.box.width == 800
    assert decision.box.height == 800
    bird_cx_px = 0.72 * w
    rel_x = (bird_cx_px - decision.box.x) / decision.box.width
    assert 0.42 <= rel_x <= 0.62
    assert "detected bird" in decision.reason


def test_u2netp_insect_mask_centers_subject_vertically_in_landscape_crop() -> None:
    """An insect segmented by U2-Net-P near the top of a 4:3 frame must be centered in 16:9."""
    h, w = 900, 1200
    img = _busy_foliage_background(h, w, seed=99)

    # U2-Net-P foreground mask isolating an insect near y=0.25 (y=225px) in 640x480 proxy
    proxy_h, proxy_w = 480, 640
    mask = np.zeros((proxy_h, proxy_w), dtype=np.float32)
    mask[85:155, 270:370] = 0.95  # Insect centered at (0.50, 0.25)

    fake_neural = NeuralSubjectResult(
        foreground_mask=mask,
        detector_tags=["u2netp_foreground"],
    )

    with patch("photo_resizer.analyzer.get_wildlife_detector") as mock_get:
        mock_get.return_value.detect.return_value = fake_neural
        # Crop 1200x900 -> 16:9 (1200x675). Ideal y to center y=225 is y=0 (rel_y = 225/675 = 0.33)
        decision = compute_smart_crop(
            img,
            CropConfig(
                aspect_ratio=AspectRatio(16, 9),
                framing_style="center_subject",
                auto_download_detectors=False,
            ),
        )

    assert decision.was_cropped is True
    assert decision.box.height == 675
    # Crop must shift all the way up near y=0 to keep insect centered & well inside frame
    assert decision.box.y <= 35
    insect_cy_px = 0.25 * h
    rel_y = (insect_cy_px - decision.box.y) / decision.box.height
    assert 0.28 <= rel_y <= 0.45


def test_yolo_detector_decodes_letterboxed_bird_box() -> None:
    """Verify _YoloAnimalDetector properly un-letterboxes a (1, 84, N) ONNX output."""

    class FakeIO:
        def __init__(self, name: str, shape: list[int]) -> None:
            self.name = name
            self.shape = shape

    class FakeYoloSession:
        def get_inputs(self) -> list[FakeIO]:
            return [FakeIO("images", [1, 3, 640, 640])]

        def get_outputs(self) -> list[FakeIO]:
            return [FakeIO("output0", [1, 84, 2])]

        def run(self, _outs: list[str], _feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            # Input 1280x640 -> letterboxed to 640x320 with pad_y=160, scale=0.5
            # Place a bird at center of letterboxed image: cx=480, cy=320, w=80, h=80
            # That maps to original x=960 (0.75), y=320 (0.50), w=160 (0.125), h=160 (0.25)
            out = np.zeros((1, 84, 2), dtype=np.float32)
            out[0, 0:4, 0] = [480.0, 320.0, 80.0, 80.0]
            out[0, 4 + 14, 0] = 0.88  # COCO class 14 = bird
            return [out]

    detector = _YoloAnimalDetector(FakeYoloSession())  # type: ignore[arg-type]
    animals, people = detector.detect(np.zeros((640, 1280, 3), dtype=np.uint8))
    assert len(people) == 0
    assert len(animals) == 1
    bird = animals[0]
    assert bird.label == "bird"
    assert abs((bird.x + bird.w * 0.5) - 0.75) < 0.02
    assert abs((bird.y + bird.h * 0.5) - 0.50) < 0.02


def test_u2net_segmenter_normalizes_foreground_mask() -> None:
    class FakeIO:
        def __init__(self, name: str, shape: list[int]) -> None:
            self.name = name
            self.shape = shape

    class FakeU2Session:
        def get_inputs(self) -> list[FakeIO]:
            return [FakeIO("input.1", [1, 3, 320, 320])]

        def get_outputs(self) -> list[FakeIO]:
            return [FakeIO("d0", [1, 1, 320, 320])]

        def run(self, _outs: list[str], _feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            mask = np.zeros((1, 1, 320, 320), dtype=np.float32)
            mask[0, 0, 100:220, 120:200] = 0.92
            return [mask]

    seg = _U2NetSegmenter(FakeU2Session())  # type: ignore[arg-type]
    sample = np.full((400, 600, 3), 60, dtype=np.uint8)
    sample[100:300, 200:400] = 200
    out = seg.segment(sample, (300, 200))
    assert out is not None
    assert out.shape == (200, 300)
    assert float(out[100, 150]) > 0.8
    assert float(out[5, 5]) == 0.0


def test_close_up_flower_that_would_clip_triggers_ai_outpaint(tmp_path: Path) -> None:
    """A tall flower in a 4:3 frame (too tall for 16:9 crop) must outpaint instead of clipping."""
    src_file = tmp_path / "closeup_flower.png"
    # 1200x900 (4:3) with distinct magenta flower color
    Image.new("RGB", (1200, 900), (220, 40, 160)).save(src_file)

    # U2-Net-P mask shows a tall flower spanning 88% of image height (y=0.06..0.94)
    # Any 16:9 crop (675px = 75% of 900px) would inevitably slice off ~15% of the petals!
    mask = np.zeros((480, 640), dtype=np.float32)
    mask[28:452, 180:460] = 0.95
    fake_neural = NeuralSubjectResult(
        foreground_mask=mask,
        detector_tags=["u2netp_foreground"],
    )

    fake_ai = Image.new("RGB", (1600, 900), (40, 110, 50))
    buf = io.BytesIO()
    fake_ai.save(buf, format="PNG")
    fake_png = buf.getvalue()

    cfg = ProcessConfig(
        source_dir=tmp_path,
        crop=CropConfig(
            aspect_ratio=AspectRatio(16, 9),
            outpaint_on_clip=True,
            auto_download_detectors=False,
        ),
        enhance=EnhanceConfig(enabled=False, gemini_api_key="test-key"),
    )

    with (
        patch("photo_resizer.analyzer.get_wildlife_detector") as mock_det,
        patch.object(GeminiOutpainter, "_call_gemini_rest", return_value=fake_png) as mock_api,
    ):
        mock_det.return_value.detect.return_value = fake_neural
        summary = process_folder(cfg)

    assert mock_api.call_count == 1
    assert summary.processed == 1
    res = summary.results[0]
    assert res.was_cropped is False
    # Expanded horizontally 1200x900 -> 1600x900 (16:9) without clipping a single petal!
    assert res.final_size == (1600, 900)
    assert "ai_outpaint_avoid_clip" in res.change_types
    assert any("Gemini AI Outpaint" in t for t in res.tools_used)


def test_bird_that_fits_cleanly_crops_offline_without_calling_ai(tmp_path: Path) -> None:
    """When an animal fits inside the crop without clipping, zero AI outpainting calls occur."""
    src_file = tmp_path / "fitting_bird.png"
    Image.new("RGB", (1200, 900), (60, 110, 60)).save(src_file)

    # Compact bird spanning y=0.30..0.60 fits comfortably inside a 75%-tall 16:9 crop
    fake_neural = NeuralSubjectResult(
        animals=[DetectedSubject(x=0.35, y=0.30, w=0.25, h=0.30, score=0.92, label="bird")],
        detector_tags=["yolo11n:bird"],
    )

    cfg = ProcessConfig(
        source_dir=tmp_path,
        crop=CropConfig(
            aspect_ratio=AspectRatio(16, 9),
            outpaint_on_clip=True,
            auto_download_detectors=False,
        ),
        enhance=EnhanceConfig(enabled=False, gemini_api_key="test-key"),
    )

    with (
        patch("photo_resizer.analyzer.get_wildlife_detector") as mock_det,
        patch.object(GeminiOutpainter, "_call_gemini_rest") as mock_api,
    ):
        mock_det.return_value.detect.return_value = fake_neural
        summary = process_folder(cfg)

    assert mock_api.call_count == 0
    res = summary.results[0]
    assert res.was_cropped is True
    assert res.final_size == (1200, 675)
    assert "aspect_ratio_crop" in res.change_types


def test_subject_centered_canvas_expansion_shifts_off_center_subject() -> None:
    # 1200x900 expanded to 16:9 (1600x900). Subject at x=0.35 (420px) should shift right
    # by offset_x = 800 - 420 = 380 so subject lands at exact center (800px of 1600px).
    cw, ch, ox, oy = compute_expanded_canvas_size(1200, 900, 16 / 9, subject_center=(0.35, 0.5))
    assert (cw, ch) == (1600, 900)
    assert ox == 380
    assert oy == 0

