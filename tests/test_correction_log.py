"""Tests for detailed per-image correction logs (TXT, CSV, JSON)."""

import csv
import json
from pathlib import Path

from PIL import Image

from photo_resizer.cli import main as cli_main
from photo_resizer.config import AspectRatio, CropConfig, EnhanceConfig, ProcessConfig
from photo_resizer.processor import process_folder


def test_correction_logs_capture_tools_and_change_types(tmp_path: Path) -> None:
    # 1. Rotated EXIF image that also needs cropping + upsampling
    rot_path = tmp_path / "rotated_small.jpg"
    exif = Image.Exif()
    exif[274] = 6  # 90 CW -> 600x800 becomes 800x600 after EXIF transpose
    Image.new("RGB", (600, 800), (80, 130, 190)).save(
        rot_path, format="JPEG", exif=exif.tobytes()
    )

    # 2. High-res image already matching 16:9 -> copied unchanged
    match_path = tmp_path / "already_16x9.png"
    Image.new("RGB", (1920, 1080), (40, 160, 80)).save(match_path)

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
        save_log=True,
        log_format="all",
    )

    summary = process_folder(cfg)
    assert summary.processed == 2

    out_dir = tmp_path / "corrected"
    txt_log = out_dir / "corrections_log.txt"
    csv_log = out_dir / "corrections_log.csv"
    json_log = out_dir / "processing_report.json"

    assert txt_log.is_file()
    assert csv_log.is_file()
    assert json_log.is_file()

    # Check TXT log includes per-image tools and change types
    txt_content = txt_log.read_text(encoding="utf-8")
    assert "rotated_small.jpg" in txt_content
    assert "exif_auto_oriented" in txt_content
    assert "aspect_ratio_crop" in txt_content
    assert "upsample_and_enhance" in txt_content
    assert "OpenCV Scene Analyzer" in txt_content
    assert "OpenCV (progressive Lanczos4" in txt_content
    assert "copied_unchanged" in txt_content

    # Check CSV log structure and values
    with csv_log.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    by_name = {row["filename"]: row for row in rows}
    rot_row = by_name["rotated_small.jpg"]
    assert "exif_auto_oriented" in rot_row["change_types"]
    assert "aspect_ratio_crop" in rot_row["change_types"]
    assert "upsample_and_enhance" in rot_row["change_types"]
    assert "Pillow (exif_transpose)" in rot_row["tools_used"]
    assert "OpenCV Scene Analyzer" in rot_row["tools_used"]
    assert "OpenCV (progressive Lanczos4" in rot_row["tools_used"]
    assert rot_row["final_size"] == "1600x900"

    unchanged_row = by_name["already_16x9.png"]
    assert unchanged_row["change_types"] == "copied_unchanged"
    assert unchanged_row["tools_used"] == "Pillow (save)"

    # Check JSON log structure
    data = json.loads(json_log.read_text(encoding="utf-8"))
    assert len(data["results"]) == 2
    rot_json = next(r for r in data["results"] if r["source_path"].endswith("rotated_small.jpg"))
    assert "exif_auto_oriented" in rot_json["change_types"]
    assert "upsample_and_enhance" in rot_json["change_types"]
    assert any("OpenCV" in t for t in rot_json["tools_used"])

    # Re-run appends skipped entries to TXT and CSV without losing earlier run history
    summary2 = process_folder(cfg)
    assert summary2.skipped == 2
    with csv_log.open("r", encoding="utf-8", newline="") as fh:
        rows2 = list(csv.DictReader(fh))
    assert len(rows2) == 4
    assert rows2[-1]["change_types"] == "skipped_already_corrected"


def test_cli_explicit_log_file_export(tmp_path: Path) -> None:
    Image.new("RGB", (1000, 1000), (120, 100, 90)).save(tmp_path / "img.png")
    custom_csv = tmp_path / "my_audit.csv"
    rc = cli_main(
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
            "--log-file",
            str(custom_csv),
        ]
    )
    assert rc == 0
    assert custom_csv.is_file()
    with custom_csv.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["filename"] == "img.png"
    assert "aspect_ratio_crop" in rows[0]["change_types"]
    assert "upsample_and_enhance" in rows[0]["change_types"]
