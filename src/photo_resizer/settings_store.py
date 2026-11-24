"""Persistent user settings storage for the Photo Resizer GUI."""

from __future__ import annotations

import base64
import ctypes
import ctypes.wintypes
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from photo_resizer.config import ASPECT_RATIO_PRESETS, DEFAULT_GEMINI_IMAGE_MODEL

VALID_ENHANCER_MODES: frozenset[str] = frozenset({"auto", "offline", "opencv", "onnx", "gemini"})
VALID_LOG_FORMATS: frozenset[str] = frozenset({"txt", "csv", "json", "all"})
VALID_ORIENTATION_MODES: frozenset[str] = frozenset(
    {"crop", "match_orientation", "blur_fill", "ai_outpaint"}
)
VALID_FRAMING_STYLES: frozenset[str] = frozenset({"center_subject", "rule_of_thirds"})


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_byte)),
    ]


def _dpapi_protect(plaintext: str) -> str | None:
    """Encrypt a secret string using Windows DPAPI (current-user scope)."""
    if sys.platform != "win32" or not plaintext:
        return None
    try:
        raw = plaintext.encode("utf-8")
        buf = (ctypes.c_byte * len(raw)).from_buffer_copy(raw)
        in_blob = _DataBlob(len(raw), buf)
        out_blob = _DataBlob()
        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32
        if not crypt32.CryptProtectData(
            ctypes.byref(in_blob),
            "PhotoResizerApiKey",
            None,
            None,
            None,
            0,
            ctypes.byref(out_blob),
        ):
            return None
        try:
            encrypted = ctypes.string_at(out_blob.pbData, out_blob.cbData)
            return base64.b64encode(encrypted).decode("ascii")
        finally:
            kernel32.LocalFree(out_blob.pbData)
    except Exception:
        return None


def _dpapi_unprotect(encoded: str) -> str | None:
    """Decrypt a Windows DPAPI current-user encrypted secret string."""
    if sys.platform != "win32" or not encoded:
        return None
    try:
        raw = base64.b64decode(encoded)
        buf = (ctypes.c_byte * len(raw)).from_buffer_copy(raw)
        in_blob = _DataBlob(len(raw), buf)
        out_blob = _DataBlob()
        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32
        if not crypt32.CryptUnprotectData(
            ctypes.byref(in_blob),
            None,
            None,
            None,
            None,
            0,
            ctypes.byref(out_blob),
        ):
            return None
        try:
            decrypted = ctypes.string_at(out_blob.pbData, out_blob.cbData)
            return decrypted.decode("utf-8")
        finally:
            kernel32.LocalFree(out_blob.pbData)
    except Exception:
        return None


@dataclass
class GuiSettings:
    """User-configurable GUI preferences persisted between launches."""

    source_dir: str = ""
    output_folder_name: str = "corrected"
    custom_output_dir: str = ""
    ratio_preset: str = "16:9 (Widescreen)"
    custom_ratio: str = "16:9"
    orientation_mode: str = "crop"
    framing_style: str = "center_subject"
    outpaint_on_clip: bool = True
    skip_existing: bool = True
    recursive: bool = False
    allow_smart_zoom: bool = False
    enhance_enabled: bool = True
    enhancer_mode: str = "auto"
    min_long_edge: int = 1920
    min_short_edge: int = 1080
    onnx_model_path: str = ""
    gemini_api_key: str = ""
    gemini_model: str = DEFAULT_GEMINI_IMAGE_MODEL
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

    framing = _coerce_str(raw.get("framing_style"), defaults.framing_style)
    if framing not in VALID_FRAMING_STYLES:
        framing = defaults.framing_style

    gemini_key = ""
    dpapi_blob = raw.get("gemini_api_key_dpapi")
    if isinstance(dpapi_blob, str) and dpapi_blob:
        gemini_key = _dpapi_unprotect(dpapi_blob) or ""
    if not gemini_key:
        gemini_key = _coerce_str(raw.get("gemini_api_key"), "")

    return GuiSettings(
        source_dir=_coerce_str(raw.get("source_dir"), defaults.source_dir),
        output_folder_name=(
            _coerce_str(raw.get("output_folder_name"), defaults.output_folder_name) or "corrected"
        ),
        custom_output_dir=_coerce_str(raw.get("custom_output_dir"), defaults.custom_output_dir),
        ratio_preset=preset,
        custom_ratio=_coerce_str(raw.get("custom_ratio"), defaults.custom_ratio) or "16:9",
        orientation_mode=orient_mode,
        framing_style=framing,
        outpaint_on_clip=_coerce_bool(raw.get("outpaint_on_clip"), defaults.outpaint_on_clip),
        skip_existing=_coerce_bool(raw.get("skip_existing"), defaults.skip_existing),
        recursive=_coerce_bool(raw.get("recursive"), defaults.recursive),
        allow_smart_zoom=_coerce_bool(raw.get("allow_smart_zoom"), defaults.allow_smart_zoom),
        enhance_enabled=_coerce_bool(raw.get("enhance_enabled"), defaults.enhance_enabled),
        enhancer_mode=mode,
        min_long_edge=_coerce_int(raw.get("min_long_edge"), defaults.min_long_edge, 320, 8192),
        min_short_edge=_coerce_int(raw.get("min_short_edge"), defaults.min_short_edge, 240, 8192),
        onnx_model_path=_coerce_str(raw.get("onnx_model_path"), defaults.onnx_model_path),
        gemini_api_key=gemini_key,
        gemini_model=(
            _coerce_str(raw.get("gemini_model"), defaults.gemini_model)
            or DEFAULT_GEMINI_IMAGE_MODEL
        ),
        save_log=_coerce_bool(raw.get("save_log"), defaults.save_log),
        log_format=log_fmt,
        window_geometry=_coerce_str(raw.get("window_geometry"), defaults.window_geometry),
    )


def save_settings(settings: GuiSettings, path: Path | None = None) -> Path:
    """Atomically save GUI settings to JSON (encrypting API key with Windows DPAPI)."""
    target = path or default_settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(settings)
    raw_key = data.pop("gemini_api_key", "")
    if raw_key:
        protected = _dpapi_protect(raw_key)
        if protected is not None:
            data["gemini_api_key_dpapi"] = protected
        else:
            data["gemini_api_key"] = raw_key
    payload = json.dumps(data, indent=2)

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
