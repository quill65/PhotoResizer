"""Persistent user settings storage for the Photo Resizer GUI."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from photo_resizer.config import ASPECT_RATIO_PRESETS

VALID_ENHANCER_MODES: frozenset[str] = frozenset({"auto", "offline", "opencv", "onnx", "gemini"})
VALID_LOG_FORMATS: frozenset[str] = frozenset({"txt", "csv", "json", "all"})
VALID_ORIENTATION_MODES: frozenset[str] = frozenset(
    {"crop", "match_orientation", "blur_fill", "ai_outpaint"}
)


@dataclass
class GuiSettings:
    """User-configurable GUI preferences persisted between launches.

    Note: API keys (such as GEMINI_API_KEY) are intentionally NOT written to
    plaintext JSON; use the GEMINI_API_KEY environment variable to persist them.
    """

    source_dir: str = ""
    output_folder_name: str = "corrected"
    custom_output_dir: str = ""
    ratio_preset: str = "16:9 (Widescreen)"
    custom_ratio: str = "16:9"
    orientation_mode: str = "crop"
    skip_existing: bool = True
    recursive: bool = False
    allow_smart_zoom: bool = False
    enhance_enabled: bool = True
    enhancer_mode: str = "auto"
    min_long_edge: int = 1920
    min_short_edge: int = 1080
    onnx_model_path: str = ""
    save_log: bool = True
    log_format: str = "txt"
    window_geometry: str = ""


def default_settings_path() -> Path:
    """Return the platform-appropriate path for settings.json."""
    override = os.environ.get("PHOTO_RESIZER_CONFIG_DIR")
    if override:
        return Path(override) / "settings.json"

    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "PhotoResizer" / "settings.json"

    return Path.home() / ".config" / "photo_resizer" / "settings.json"


def _coerce_bool(val: object, default: bool) -> bool:
    return val if isinstance(val, bool) else default


def _coerce_str(val: object, default: str) -> str:
    return val if isinstance(val, str) else default


def _coerce_int(val: object, default: int, min_val: int, max_val: int) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        return default
    return max(min_val, min(max_val, val))


def load_settings(path: Path | None = None) -> GuiSettings:
    """Load saved GUI settings, falling back safely to defaults on any error."""
    target = path or default_settings_path()
    defaults = GuiSettings()
    if not target.is_file():
        return defaults

    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return defaults

    if not isinstance(raw, dict):
        return defaults

    valid_presets = set(ASPECT_RATIO_PRESETS.keys()) | {"Custom"}
    preset = _coerce_str(raw.get("ratio_preset"), defaults.ratio_preset)
    if preset not in valid_presets:
        preset = defaults.ratio_preset

    mode = _coerce_str(raw.get("enhancer_mode"), defaults.enhancer_mode)
    if mode not in VALID_ENHANCER_MODES:
        mode = defaults.enhancer_mode

    log_fmt = _coerce_str(raw.get("log_format"), defaults.log_format)
    if log_fmt not in VALID_LOG_FORMATS:
        log_fmt = defaults.log_format

    orient_mode = _coerce_str(raw.get("orientation_mode"), defaults.orientation_mode)
    if orient_mode not in VALID_ORIENTATION_MODES:
        orient_mode = defaults.orientation_mode

    return GuiSettings(
        source_dir=_coerce_str(raw.get("source_dir"), defaults.source_dir),
        output_folder_name=(
            _coerce_str(raw.get("output_folder_name"), defaults.output_folder_name) or "corrected"
        ),
        custom_output_dir=_coerce_str(raw.get("custom_output_dir"), defaults.custom_output_dir),
        ratio_preset=preset,
        custom_ratio=_coerce_str(raw.get("custom_ratio"), defaults.custom_ratio) or "16:9",
        orientation_mode=orient_mode,
        skip_existing=_coerce_bool(raw.get("skip_existing"), defaults.skip_existing),
        recursive=_coerce_bool(raw.get("recursive"), defaults.recursive),
        allow_smart_zoom=_coerce_bool(raw.get("allow_smart_zoom"), defaults.allow_smart_zoom),
        enhance_enabled=_coerce_bool(raw.get("enhance_enabled"), defaults.enhance_enabled),
        enhancer_mode=mode,
        min_long_edge=_coerce_int(raw.get("min_long_edge"), defaults.min_long_edge, 320, 8192),
        min_short_edge=_coerce_int(raw.get("min_short_edge"), defaults.min_short_edge, 240, 8192),
        onnx_model_path=_coerce_str(raw.get("onnx_model_path"), defaults.onnx_model_path),
        save_log=_coerce_bool(raw.get("save_log"), defaults.save_log),
        log_format=log_fmt,
        window_geometry=_coerce_str(raw.get("window_geometry"), defaults.window_geometry),
    )


def save_settings(settings: GuiSettings, path: Path | None = None) -> Path:
    """Atomically save GUI settings to JSON."""
    target = path or default_settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(settings), indent=2)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix="settings_",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(payload)
        tmp_path.replace(target)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise

    return target
