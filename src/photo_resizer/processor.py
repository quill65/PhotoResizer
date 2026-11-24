"""Batch folder scanner and idempotent image processor."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from PIL import Image

from photo_resizer.config import SUPPORTED_EXTENSIONS, ProcessConfig
from photo_resizer.cropper import crop_pil_image, load_oriented_image
from photo_resizer.enhancer import TieredEnhancer

StatusType = Literal["processed", "skipped", "failed"]


@dataclass
class ImageResult:
    """Outcome for a single image file in a batch run."""

    source_path: Path
    output_path: Path
    status: StatusType
    original_size: tuple[int, int] | None = None
    cropped_size: tuple[int, int] | None = None
    final_size: tuple[int, int] | None = None
    was_cropped: bool = False
    was_enhanced: bool = False
    enhancer_used: str = "none"
    message: str = ""

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["source_path"] = str(self.source_path)
        data["output_path"] = str(self.output_path)
        return data


@dataclass
class BatchSummary:
    """Aggregate summary of a folder processing run."""

    source_dir: Path
    output_dir: Path
    total_found: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    cancelled: bool = False
    results: list[ImageResult] = field(default_factory=list)


ProgressCallback = Callable[[int, int, ImageResult], None]


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def discover_images(source_dir: Path, output_dir: Path, recursive: bool = False) -> list[Path]:
    """Find all supported images in source_dir while excluding the output directory."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    iterator = source_dir.rglob("*") if recursive else source_dir.iterdir()
    images: list[Path] = []
    for entry in iterator:
        if not entry.is_file():
            continue
        if _is_inside(entry, output_dir):
            continue
        if entry.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(entry)

    return sorted(images, key=lambda p: str(p).lower())


def target_output_path(source_path: Path, source_dir: Path, output_dir: Path) -> Path:
    """Compute the destination path inside output_dir for a source image."""
    rel = source_path.relative_to(source_dir)
    return output_dir / rel


def already_corrected(output_path: Path) -> bool:
    """Return True if the corrected output image already exists and is non-empty."""
    return output_path.is_file() and output_path.stat().st_size > 0


def _save_image(
    image: Image.Image,
    output_path: Path,
    jpeg_quality: int = 94,
    exif_bytes: bytes | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()
    save_kwargs: dict[str, object] = {}

    to_save = image
    if ext in {".jpg", ".jpeg"}:
        if to_save.mode != "RGB":
            to_save = to_save.convert("RGB")
        save_kwargs["quality"] = max(1, min(100, jpeg_quality))
        save_kwargs["subsampling"] = 0
        save_kwargs["optimize"] = True
        if exif_bytes:
            save_kwargs["exif"] = exif_bytes
    elif ext == ".webp":
        save_kwargs["quality"] = max(1, min(100, jpeg_quality))
        save_kwargs["method"] = 6
        if exif_bytes:
            save_kwargs["exif"] = exif_bytes
    elif ext == ".png":
        save_kwargs["optimize"] = True
    elif ext in {".tif", ".tiff"}:
        save_kwargs["compression"] = "tiff_deflate"

    to_save.save(output_path, **save_kwargs)


def process_folder(
    config: ProcessConfig,
    on_progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> BatchSummary:
    """Process all images in config.source_dir, skipping any already in the corrected folder."""
    source_dir = config.source_dir.resolve()
    output_dir = config.resolved_output_dir().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = discover_images(source_dir, output_dir, recursive=config.recursive)
    summary = BatchSummary(
        source_dir=source_dir,
        output_dir=output_dir,
        total_found=len(image_paths),
    )

    enhancer = TieredEnhancer(config.enhance)

    for idx, src_path in enumerate(image_paths, start=1):
        if cancel_event is not None and cancel_event.is_set():
            summary.cancelled = True
            break

        out_path = target_output_path(src_path, source_dir, output_dir)

        if not config.overwrite and already_corrected(out_path):
            res = ImageResult(
                source_path=src_path,
                output_path=out_path,
                status="skipped",
                message="Already corrected (present in output folder)",
            )
            summary.skipped += 1
            summary.results.append(res)
            if on_progress is not None:
                on_progress(idx, len(image_paths), res)
            continue

        try:
            oriented, exif_bytes = load_oriented_image(src_path)
            orig_size = oriented.size

            cropped_img, crop_decision = crop_pil_image(oriented, config.crop)
            cropped_size = cropped_img.size

            enhance_result = enhancer.enhance_if_needed(
                cropped_img,
                target_ratio=config.crop.aspect_ratio.value,
            )
            final_img = enhance_result.image
            final_size = final_img.size

            _save_image(
                final_img,
                out_path,
                jpeg_quality=config.jpeg_quality,
                exif_bytes=exif_bytes,
            )

            msg_parts = [crop_decision.reason]
            if enhance_result.was_enhanced:
                msg_parts.append(enhance_result.note)
            elif config.enhance.enabled:
                msg_parts.append("no upscaling needed")

            res = ImageResult(
                source_path=src_path,
                output_path=out_path,
                status="processed",
                original_size=orig_size,
                cropped_size=cropped_size,
                final_size=final_size,
                was_cropped=crop_decision.was_cropped,
                was_enhanced=enhance_result.was_enhanced,
                enhancer_used=enhance_result.backend,
                message=" | ".join(msg_parts),
            )
            summary.processed += 1
        except Exception as exc:
            res = ImageResult(
                source_path=src_path,
                output_path=out_path,
                status="failed",
                message=f"Error: {exc}",
            )
            summary.failed += 1

        summary.results.append(res)
        if on_progress is not None:
            on_progress(idx, len(image_paths), res)

    if config.write_report:
        _write_report(summary, config)

    return summary


def _write_report(summary: BatchSummary, config: ProcessConfig) -> None:
    report_path = summary.output_dir / "processing_report.json"
    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_dir": str(summary.source_dir),
        "output_dir": str(summary.output_dir),
        "aspect_ratio": config.crop.aspect_ratio.display_name,
        "enhancer_mode": config.enhance.mode,
        "min_long_edge": config.enhance.min_long_edge,
        "min_short_edge": config.enhance.min_short_edge,
        "total_found": summary.total_found,
        "processed": summary.processed,
        "skipped": summary.skipped,
        "failed": summary.failed,
        "cancelled": summary.cancelled,
        "results": [r.to_json_dict() for r in summary.results],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
