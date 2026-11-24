"""Structured correction log writers (TXT, CSV, JSON) for batch runs."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from photo_resizer.config import ProcessConfig
    from photo_resizer.processor import BatchSummary, ImageResult


CSV_COLUMNS: tuple[str, ...] = (
    "timestamp_utc",
    "filename",
    "status",
    "original_size",
    "cropped_size",
    "final_size",
    "crop_box",
    "upscale_factor",
    "change_types",
    "tools_used",
    "faces_detected",
    "horizon_y_ratio",
    "details",
    "source_path",
    "output_path",
)


def _fmt_size(size: tuple[int, int] | None) -> str:
    return f"{size[0]}x{size[1]}" if size is not None else ""


def _fmt_crop_box(box: tuple[int, int, int, int] | None) -> str:
    if box is None:
        return ""
    x, y, w, h = box
    return f"x={x},y={y},w={w},h={h}"


def format_text_log(summary: BatchSummary, config: ProcessConfig) -> str:
    """Format a human-readable correction log section describing every image."""
    now_iso = datetime.now(UTC).isoformat()
    enh_label = config.enhance.mode if config.enhance.enabled else "disabled"
    lines: list[str] = [
        "=" * 78,
        f"Photo Resizer Correction Log — {now_iso}",
        f"Source Folder : {summary.source_dir}",
        f"Output Folder : {summary.output_dir}",
        f"Aspect Ratio  : {config.crop.aspect_ratio.display_name} "
        f"(orientation={config.crop.orientation_mode}, "
        f"smart_zoom={'on' if config.crop.allow_smart_zoom else 'off'})",
        f"Enhancement   : {enh_label} "
        f"(min {config.enhance.min_long_edge}x{config.enhance.min_short_edge})",
        f"Summary       : Found {summary.total_found} | Processed {summary.processed} | "
        f"Skipped {summary.skipped} | Failed {summary.failed}"
        + (" | CANCELLED" if summary.cancelled else ""),
        "=" * 78,
    ]

    for result in summary.results:
        lines.extend(_format_single_text_entry(result))

    lines.append("")
    return "\n".join(lines)


def _format_single_text_entry(result: ImageResult) -> list[str]:
    entry = [f"- {result.source_path.name} [{result.status.upper()}]"]
    if result.original_size is not None:
        dim_chain = _fmt_size(result.original_size)
        if result.cropped_size is not None and result.cropped_size != result.original_size:
            dim_chain += f" -> cropped {_fmt_size(result.cropped_size)}"
        if result.final_size is not None and result.final_size != result.cropped_size:
            dim_chain += (
                f" -> final {_fmt_size(result.final_size)} ({result.upscale_factor:.2f}x)"
            )
        elif result.final_size is not None and result.cropped_size == result.original_size:
            dim_chain += f" -> final {_fmt_size(result.final_size)}"
        entry.append(f"    Dimensions : {dim_chain}")

    if result.crop_box is not None and result.was_cropped:
        entry.append(f"    Crop Box   : {_fmt_crop_box(result.crop_box)}")

    changes_str = ", ".join(result.change_types) if result.change_types else "none"
    tools_str = "; ".join(result.tools_used) if result.tools_used else "none"
    entry.append(f"    Changes    : {changes_str}")
    entry.append(f"    Tools Used : {tools_str}")
    if result.faces_detected > 0 or result.horizon_y_ratio is not None:
        scene_bits: list[str] = []
        if result.faces_detected > 0:
            scene_bits.append(f"faces={result.faces_detected}")
        if result.horizon_y_ratio is not None:
            scene_bits.append(f"horizon={result.horizon_y_ratio:.0%}")
        entry.append(f"    Scene Info : {', '.join(scene_bits)}")
    entry.append(f"    Details    : {result.message}")
    return entry


def write_text_log(
    summary: BatchSummary,
    config: ProcessConfig,
    path: Path,
    *,
    append: bool = True,
) -> Path:
    """Write or append the human-readable text correction log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = format_text_log(summary, config)
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, encoding="utf-8") as fh:
        fh.write(content)
    return path


def write_csv_log(
    summary: BatchSummary,
    config: ProcessConfig,
    path: Path,
    *,
    append: bool = True,
) -> Path:
    """Write or append a spreadsheet-friendly CSV correction log."""
    del config  # Reserved for future metadata columns
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (append and path.exists() and path.stat().st_size > 0)
    mode = "a" if append and path.exists() else "w"

    with path.open(mode, encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for r in summary.results:
            writer.writerow(
                {
                    "timestamp_utc": r.processed_at_utc,
                    "filename": r.source_path.name,
                    "status": r.status,
                    "original_size": _fmt_size(r.original_size),
                    "cropped_size": _fmt_size(r.cropped_size),
                    "final_size": _fmt_size(r.final_size),
                    "crop_box": _fmt_crop_box(r.crop_box) if r.was_cropped else "",
                    "upscale_factor": f"{r.upscale_factor:.2f}" if r.was_enhanced else "1.00",
                    "change_types": "; ".join(r.change_types),
                    "tools_used": "; ".join(r.tools_used),
                    "faces_detected": r.faces_detected,
                    "horizon_y_ratio": (
                        f"{r.horizon_y_ratio:.3f}" if r.horizon_y_ratio is not None else ""
                    ),
                    "details": r.message,
                    "source_path": str(r.source_path),
                    "output_path": str(r.output_path),
                }
            )
    return path


def build_json_payload(summary: BatchSummary, config: ProcessConfig) -> dict[str, object]:
    """Build structured JSON report dictionary for a batch run."""
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_dir": str(summary.source_dir),
        "output_dir": str(summary.output_dir),
        "aspect_ratio": config.crop.aspect_ratio.display_name,
        "orientation_mode": config.crop.orientation_mode,
        "allow_smart_zoom": config.crop.allow_smart_zoom,
        "enhancer_mode": config.enhance.mode if config.enhance.enabled else "disabled",
        "min_long_edge": config.enhance.min_long_edge,
        "min_short_edge": config.enhance.min_short_edge,
        "total_found": summary.total_found,
        "processed": summary.processed,
        "skipped": summary.skipped,
        "failed": summary.failed,
        "cancelled": summary.cancelled,
        "results": [r.to_json_dict() for r in summary.results],
    }


def write_json_log(summary: BatchSummary, config: ProcessConfig, path: Path) -> Path:
    """Write structured JSON correction log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_json_payload(summary, config)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def export_log_to_path(
    summary: BatchSummary,
    config: ProcessConfig,
    path: Path,
    *,
    append: bool = False,
) -> Path:
    """Export a batch summary to a specific path, inferring format from file extension."""
    ext = path.suffix.lower()
    if ext == ".csv":
        return write_csv_log(summary, config, path, append=append)
    if ext == ".json":
        return write_json_log(summary, config, path)
    return write_text_log(summary, config, path, append=append)


def save_batch_logs(summary: BatchSummary, config: ProcessConfig) -> list[Path]:
    """Save configured correction log(s) and JSON report for a completed batch."""
    written: list[Path] = []
    report_path = summary.output_dir / "processing_report.json"

    if config.log_file is not None:
        written.append(export_log_to_path(summary, config, config.log_file, append=True))
    elif config.save_log:
        fmt = config.log_format
        if fmt in ("txt", "all"):
            written.append(
                write_text_log(
                    summary, config, summary.output_dir / "corrections_log.txt", append=True
                )
            )
        if fmt in ("csv", "all"):
            written.append(
                write_csv_log(
                    summary, config, summary.output_dir / "corrections_log.csv", append=True
                )
            )
        if fmt in ("json", "all"):
            written.append(write_json_log(summary, config, report_path))

    if config.write_report and report_path not in written:
        written.append(write_json_log(summary, config, report_path))

    return written
