"""Tests for persistent GUI settings storage."""

import json
from pathlib import Path

from photo_resizer.settings_store import GuiSettings, load_settings, save_settings


def test_load_settings_returns_defaults_when_missing(tmp_path: Path) -> None:
    loaded = load_settings(tmp_path / "missing.json")
    assert loaded == GuiSettings()


def test_save_and_reload_settings_round_trip(tmp_path: Path) -> None:
    cfg_file = tmp_path / "settings.json"
    original = GuiSettings(
        source_dir=r"C:\Photos\Vacation",
        output_folder_name="cropped_16x9",
        custom_output_dir=r"C:\Photos\Out",
        ratio_preset="4:5 (Portrait Social)",
        custom_ratio="4:5",
        skip_existing=False,
        recursive=True,
        allow_smart_zoom=True,
        enhance_enabled=True,
        enhancer_mode="onnx",
        min_long_edge=2560,
        min_short_edge=1440,
        onnx_model_path=r"C:\models\realesr-general-x4v3.onnx",
        window_geometry="900x700+100+80",
    )

    save_settings(original, cfg_file)
    assert cfg_file.is_file()

    # Ensure no secret/API key field is written to plaintext JSON
    raw_text = cfg_file.read_text(encoding="utf-8")
    assert "api_key" not in raw_text.lower()

    reloaded = load_settings(cfg_file)
    assert reloaded == original


def test_load_settings_handles_corrupt_and_invalid_values(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not valid json", encoding="utf-8")
    assert load_settings(corrupt) == GuiSettings()

    partial = tmp_path / "partial.json"
    partial.write_text(
        json.dumps(
            {
                "source_dir": r"C:\Pictures",
                "ratio_preset": "NonExistentPreset",
                "enhancer_mode": "unknown_backend",
                "min_long_edge": "not_an_int",
                "min_short_edge": 999999,
                "allow_smart_zoom": True,
            }
        ),
        encoding="utf-8",
    )
    loaded = load_settings(partial)
    assert loaded.source_dir == r"C:\Pictures"
    assert loaded.allow_smart_zoom is True
    # Invalid values fall back safely
    assert loaded.ratio_preset == "16:9 (Widescreen)"
    assert loaded.enhancer_mode == "auto"
    assert loaded.min_long_edge == 1920
    assert loaded.min_short_edge == 8192  # Clamped to max valid bound
