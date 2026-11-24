"""Batch folder scanner and idempotent image processor."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from PIL import Image

from photo_resizer.config import SUPPORTED_EXTENSIONS, AspectRatio, ProcessConfig
from photo_resizer.correction_log import save_batch_logs
from photo_resizer.cropper import crop_pil_image, load_oriented_image_with_meta
from photo_resizer.enhancer import TieredEnhancer
from photo_resizer.outpainter import (
    expand_canvas_ai_outpaint,
    expand_canvas_blur_fill,
    is_orientation_mismatch,
)

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
    crop_box: tuple[int, int, int, int] | None = None
    was_cropped: bool = False
    was_enhanced: bool = False
    enhancer_used: str = "none"
    upscale_factor: float = 1.0
    change_types: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    faces_detected: int = 0
    horizon_y_ratio: float | None = None
    processed_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
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
    saved_log_paths: list[Path] = field(default_factory=list)


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
                change_types=["skipped_already_corrected"],
                message="Already corrected (present in output folder)",
            )
            summary.skipped += 1
            summary.results.append(res)
            if on_progress is not None:
                on_progress(idx, len(image_paths), res)
            continue

        try:
            oriented, exif_bytes, was_exif_rotated = load_oriented_image_with_meta(src_path)
            orig_size = oriented.size

            change_types: list[str] = []
            tools_used: list[str] = []
            if was_exif_rotated:
                change_types.append("exif_auto_oriented")
                tools_used.append("Pillow (exif_transpose)")

            target_ratio_obj = config.crop.aspect_ratio
            orient_mode = config.crop.orientation_mode
            mismatched = is_orientation_mismatch(
                orig_size[0], orig_size[1], target_ratio_obj.value
            )

            was_cropped = False
            crop_box_tuple: tuple[int, int, int, int] | None = None
            faces_count = 0
            horizon_rel: float | None = None
            effective_ratio_val = target_ratio_obj.value
            framing_note = ""

            if mismatched and orient_mode == "blur_fill":
                expanded = expand_canvas_blur_fill(oriented, target_ratio_obj.value)
                framed_img = expanded.image
                change_types.append("blurred_backdrop_fill")
                tools_used.append(expanded.tool_detail)
                framing_note = expanded.note
            elif mismatched and orient_mode == "ai_outpaint":
                expanded = expand_canvas_ai_outpaint(
                    oriented, target_ratio_obj.value, config.enhance
                )
                framed_img = expanded.image
                if expanded.method == "ai_outpaint":
                    change_types.append("ai_outpaint_background")
                else:
                    change_types.append("blurred_backdrop_fill")
                tools_used.append(expanded.tool_detail)
                framing_note = expanded.note
            else:
                effective_crop_cfg = config.crop
                if mismatched and orient_mode == "match_orientation":
                    flipped = AspectRatio(target_ratio_obj.height, target_ratio_obj.width)
                    effective_crop_cfg = replace(config.crop, aspect_ratio=flipped)
                    effective_ratio_val = flipped.value
                    change_types.append("orientation_matched")

                framed_img, crop_decision = crop_pil_image(oriented, effective_crop_cfg)
                faces_count = (
                    len(crop_decision.analysis.faces)
                    if crop_decision.analysis is not None
                    else 0
                )
                horizon_rel = crop_decision.horizon_y_ratio

                # Record offline detector tools whenever scene analysis ran
                if crop_decision.analysis is not None and crop_decision.analysis.detector_tags:
                    tools_used.append(
                        "ONNX Wildlife Detector ("
                        + ", ".join(crop_decision.analysis.detector_tags)
                        + ")"
                    )

                # If cropping would slice off part of a detected animal, flower, or face,
                # expand the canvas via AI outpainting (with offline blur-fill fallback)
                # instead of clipping the subject!
                if (
                    crop_decision.was_cropped
                    and crop_decision.would_clip_subject
                    and config.crop.outpaint_on_clip
                ):
                    subj_center = (
                        crop_decision.analysis.primary_center
                        if (
                            crop_decision.analysis is not None
                            and config.crop.framing_style == "center_subject"
                        )
                        else None
                    )
                    clip_pct = crop_decision.subject_clipped_fraction
                    expanded = expand_canvas_ai_outpaint(
                        oriented,
                        effective_ratio_val,
                        config.enhance,
                        subject_center=subj_center,
                        reason_prefix=f"subject clip avoided ({clip_pct:.0%} would be cut)",
                    )
                    framed_img = expanded.image
                    was_cropped = False
                    crop_box_tuple = None
                    if expanded.method == "ai_outpaint":
                        change_types.append("ai_outpaint_avoid_clip")
                    else:
                        change_types.append("blurred_backdrop_avoid_clip")
                    tools_used.append(expanded.tool_detail)
                    framing_note = expanded.note
                else:
                    was_cropped = crop_decision.was_cropped
                    box = crop_decision.box
                    crop_box_tuple = (box.x, box.y, box.width, box.height)
                    framing_note = crop_decision.reason
                    if mismatched and orient_mode == "match_orientation":
                        ratio_name = effective_crop_cfg.aspect_ratio.display_name
                        framing_note = f"matched orientation ({ratio_name}); {framing_note}"

                    if crop_decision.was_cropped:
                        change_types.append("aspect_ratio_crop")
                        if crop_decision.was_smart_zoomed:
                            change_types.append("smart_zoom_reframe")
                        signals: list[str] = []
                        if faces_count > 0:
                            signals.append("haar_faces")
                        signals.append("spectral+lab_saliency")
                        if horizon_rel is not None:
                            signals.append("hough_horizon")
                        tools_used.append(f"OpenCV Scene Analyzer ({', '.join(signals)})")
                        tools_used.append("Pillow (crop)")

            cropped_size = framed_img.size

            enhance_result = enhancer.enhance_if_needed(
                framed_img,
                target_ratio=effective_ratio_val,
            )
            final_img = enhance_result.image
            final_size = final_img.size

            _save_image(
                final_img,
                out_path,
                jpeg_quality=config.jpeg_quality,
                exif_bytes=exif_bytes,
            )

            if enhance_result.was_enhanced:
                change_types.append("upsample_and_enhance")
                tools_used.append(enhance_result.tool_detail or enhance_result.backend)

            if not change_types or change_types == ["orientation_matched"]:
                change_types.append("copied_unchanged")
                tools_used.append("Pillow (save)")

            msg_parts = [framing_note]
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
                crop_box=crop_box_tuple,
                was_cropped=was_cropped,
                was_enhanced=enhance_result.was_enhanced,
                enhancer_used=enhance_result.backend,
                upscale_factor=enhance_result.scale_factor,
                change_types=change_types,
                tools_used=tools_used,
                faces_detected=faces_count,
                horizon_y_ratio=horizon_rel,
                message=" | ".join(msg_parts),
            )
            summary.processed += 1
        except Exception as exc:
            res = ImageResult(
                source_path=src_path,
                output_path=out_path,
                status="failed",
                change_types=["failed"],
                message=f"Error: {exc}",
            )
            summary.failed += 1

        summary.results.append(res)
        if on_progress is not None:
            on_progress(idx, len(image_paths), res)

    summary.saved_log_paths = save_batch_logs(summary, config)
    return summary
