"""Tests for configuration models and aspect ratio parsing."""

from pathlib import Path

import pytest

from photo_resizer.config import AspectRatio, ProcessConfig


def test_aspect_ratio_parse_variants() -> None:
    assert AspectRatio.parse("16:9").value == pytest.approx(16 / 9)
    assert AspectRatio.parse("4x3").value == pytest.approx(4 / 3)
    assert AspectRatio.parse("3/2").value == pytest.approx(1.5)
    assert AspectRatio.parse("square").value == pytest.approx(1.0)
    assert AspectRatio.parse("16:9 (Widescreen)").value == pytest.approx(16 / 9)
    assert AspectRatio.parse("2.35").value == pytest.approx(2.35)


def test_aspect_ratio_parse_invalid() -> None:
    with pytest.raises(ValueError):
        AspectRatio.parse("invalid")
    with pytest.raises(ValueError):
        AspectRatio.parse("0:9")
    with pytest.raises(ValueError):
        AspectRatio.parse("-4:3")


def test_resolved_output_dir_defaults_to_corrected_subfolder(tmp_path: Path) -> None:
    cfg = ProcessConfig(source_dir=tmp_path)
    assert cfg.resolved_output_dir() == tmp_path / "corrected"

    custom_out = tmp_path / "custom_out"
    cfg_custom = ProcessConfig(source_dir=tmp_path, output_dir=custom_out)
    assert cfg_custom.resolved_output_dir() == custom_out
