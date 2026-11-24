"""End-to-end tests for batch folder processing, idempotent skips, and CLI."""

import json
from pathlib import Path

from PIL import Image

from photo_resizer.cli import main as cli_main
from photo_resizer.config import AspectRatio, CropConfig, EnhanceConfig, ProcessConfig
from photo_resizer.gui import HELP_SECTIONS
from photo_resizer.processor import process_folder


def _make_sample_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    img = Image.new("RGB", size, color=color)
    img.save(path)


def test_batch_process_and_idempotent_rerun(tmp_path: Path) -> None:
    img1 = tmp_path / "wide_highres.jpg"
    img2 = tmp_path / "tall_lowres.png"
    _make_sample_image(img1, (2400, 1800), (60, 120, 180))  # 4:3 -> crop to 16:9 (2400x1350)
    _make_sample_image(img2, (800, 1000), (180, 90, 60))  # 4:5 -> crop to 16:9 (800x450) -> upscale

    cfg = ProcessConfig(
        source_dir=tmp_path,
        crop=CropConfig(aspect_ratio=AspectRatio(16, 9)),
        enhance=EnhanceConfig(
            enabled=True,
            mode="opencv",
            min_long_edge=1600,
            min_short_edge=900,
            auto_download_onnx=False,
        ),
    )

    # First run: processes both images into <tmp_path>/corrected
    summary1 = process_folder(cfg)
    assert summary1.total_found == 2
    assert summary1.processed == 2
    assert summary1.skipped == 0
    assert summary1.failed == 0

    out_dir = tmp_path / "corrected"
    assert (out_dir / "wide_highres.jpg").is_file()
    assert (out_dir / "tall_lowres.png").is_file()
    assert (out_dir / "processing_report.json").is_file()

    with Image.open(out_dir / "wide_highres.jpg") as im1:
        assert im1.size == (2400, 1350)  # Cropped, high enough res -> not upscaled
    with Image.open(out_dir / "tall_lowres.png") as im2:
        assert im2.size == (1600, 900)  # Cropped 800x450 -> upscaled 2x to 1600x900

    report = json.loads((out_dir / "processing_report.json").read_text(encoding="utf-8"))
    assert report["processed"] == 2

    # Second run on same folder: skips already-corrected images and never re-scans corrected/
    summary2 = process_folder(cfg)
    assert summary2.total_found == 2
    assert summary2.processed == 0
    assert summary2.skipped == 2
    assert summary2.failed == 0

    # Add a third brand-new image -> only the new image is processed
    img3 = tmp_path / "new_photo.jpg"
    _make_sample_image(img3, (2000, 1500), (90, 160, 90))
    summary3 = process_folder(cfg)
    assert summary3.total_found == 3
    assert summary3.processed == 1
    assert summary3.skipped == 2
    assert (out_dir / "new_photo.jpg").is_file()


def test_cli_end_to_end(tmp_path: Path) -> None:
    _make_sample_image(tmp_path / "sample.png", (1000, 1000), (110, 130, 150))
    exit_code = cli_main(
        [
            str(tmp_path),
            "--ratio",
            "16:9",
            "--enhancer",
            "opencv",
            "--min-long-edge",
            "1280",
            "--min-short-edge",
            "720",
        ]
    )
    assert exit_code == 0
    out_file = tmp_path / "corrected" / "sample.png"
    assert out_file.is_file()
    with Image.open(out_file) as out_img:
        assert out_img.size == (1280, 720)


def test_gui_help_sections_cover_key_options() -> None:
    flattened = " ".join(
        f"{section} {opt} {desc}"
        for section, items in HELP_SECTIONS
        for opt, desc in items
    ).lower()
    for keyword in (
        "smart zoom",
        "onnx",
        "gemini",
        "opencv",
        "skip images",
        "aspect ratio",
        "aistudio.google.com",
        "google account",
    ):
        assert keyword in flattened

