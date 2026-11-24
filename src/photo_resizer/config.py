"""Configuration models and presets for Photo Resizer."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

EnhancerMode = Literal["auto", "offline", "opencv", "onnx", "gemini"]
LogFormat = Literal["txt", "csv", "json", "all"]
OrientationMode = Literal["crop", "match_orientation", "blur_fill", "ai_outpaint"]

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)


@dataclass(frozen=True)
class AspectRatio:
    """Target aspect ratio expressed as width : height."""

    width: float
    height: float
    label: str = ""

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Aspect ratio must be positive, got {self.width}:{self.height}")

    @property
    def value(self) -> float:
        return self.width / self.height

    @property
    def display_name(self) -> str:
        if self.label:
            return self.label
        w = int(self.width) if float(self.width).is_integer() else self.width
        h = int(self.height) if float(self.height).is_integer() else self.height
        return f"{w}:{h}"

    @classmethod
    def parse(cls, text: str) -> AspectRatio:
        """Parse strings like '16:9', '4x3', '3/2', '1.777', or 'square'."""
        cleaned = text.strip().lower()
        if cleaned in {"square", "1:1"}:
            return cls(1.0, 1.0, "1:1")

        # Strip optional preset description in parentheses, e.g. "16:9 (Widescreen)"
        if "(" in cleaned:
            cleaned = cleaned.split("(", 1)[0].strip()

        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[:/x]\s*(\d+(?:\.\d+)?)", cleaned)
        if match:
            w = float(match.group(1))
            h = float(match.group(2))
            return cls(w, h)

        try:
            ratio = float(cleaned)
        except ValueError as exc:
            raise ValueError(
                f"Invalid aspect ratio '{text}'. Use formats like '16:9', '4:3', or '1.5'."
            ) from exc

        if ratio <= 0:
            raise ValueError(f"Aspect ratio must be positive, got '{text}'.")
        return cls(ratio, 1.0, f"{ratio:g}:1")


ASPECT_RATIO_PRESETS: dict[str, AspectRatio] = {
    "16:9 (Widescreen)": AspectRatio(16, 9, "16:9"),
    "3:2 (Classic Photo)": AspectRatio(3, 2, "3:2"),
    "4:3 (Standard)": AspectRatio(4, 3, "4:3"),
    "5:4 (Print 8x10)": AspectRatio(5, 4, "5:4"),
    "1:1 (Square)": AspectRatio(1, 1, "1:1"),
    "4:5 (Portrait Social)": AspectRatio(4, 5, "4:5"),
    "2:3 (Portrait Photo)": AspectRatio(2, 3, "2:3"),
    "9:16 (Vertical)": AspectRatio(9, 16, "9:16"),
}


@dataclass
class CropConfig:
    """Smart cropping and scene analysis parameters."""

    aspect_ratio: AspectRatio = field(default_factory=lambda: AspectRatio(16, 9, "16:9"))
    aspect_tolerance: float = 0.01
    orientation_mode: OrientationMode = "crop"
    allow_smart_zoom: bool = False
    min_crop_fraction: float = 0.85
    candidate_steps: int = 41
    analysis_max_dim: int = 640


@dataclass
class EnhanceConfig:
    """Tiered upsampling and enhancement configuration."""

    enabled: bool = True
    mode: EnhancerMode = "auto"
    min_long_edge: int = 1920
    min_short_edge: int = 1080
    max_upscale_factor: float = 4.0
    mild_upscale_threshold: float = 1.4
    onnx_model_path: Path | None = None
    auto_download_onnx: bool = True
    onnx_tile_size: int = 512
    onnx_tile_overlap: int = 32
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash-image"
    gemini_extreme_threshold: float = 2.5
    use_gemini_for_extreme_in_auto: bool = False

    def resolved_gemini_api_key(self) -> str | None:
        return self.gemini_api_key or os.environ.get("GEMINI_API_KEY")


@dataclass
class ProcessConfig:
    """Top-level configuration for batch processing an image folder."""

    source_dir: Path
    output_dir: Path | None = None
    output_folder_name: str = "corrected"
    crop: CropConfig = field(default_factory=CropConfig)
    enhance: EnhanceConfig = field(default_factory=EnhanceConfig)
    jpeg_quality: int = 94
    overwrite: bool = False
    recursive: bool = False
    write_report: bool = True
    save_log: bool = True
    log_format: LogFormat = "txt"
    log_file: Path | None = None

    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return self.output_dir
        return self.source_dir / self.output_folder_name
