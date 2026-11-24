"""Command-line interface for Photo Resizer (with optional native folder picker)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from photo_resizer.config import (
    AspectRatio,
    CropConfig,
    EnhanceConfig,
    ProcessConfig,
)
from photo_resizer.processor import ImageResult, process_folder


def pick_folder_dialog(title: str = "Select Original Image Folder") -> Path | None:
    """Open a native folder selection dialog using tkinter."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(title=title)
    finally:
        root.destroy()
    return Path(selected) if selected else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photo-resizer-cli",
        description=(
            "Smart offline-first photo aspect-ratio cropper and tiered upsampler. "
            "If source_dir is omitted, a native folder picker dialog will open."
        ),
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Folder containing original images (opens folder picker if omitted).",
    )
    parser.add_argument(
        "-r",
        "--ratio",
        default="16:9",
        help="Target aspect ratio, e.g. 16:9, 4:3, 3:2, 1:1, 4:5, 9:16 (default: 16:9).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Explicit output directory (default: <source_dir>/corrected).",
    )
    parser.add_argument(
        "--output-name",
        default="corrected",
        help="Subfolder name inside source_dir when --output is not set (default: corrected).",
    )
    parser.add_argument(
        "--enhancer",
        choices=["auto", "offline", "opencv", "onnx", "gemini"],
        default="auto",
        help="Upsampling backend mode when a crop falls below minimum resolution (default: auto).",
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Disable automatic upsampling/enhancement of low-resolution crops.",
    )
    parser.add_argument(
        "--min-long-edge",
        type=int,
        default=1920,
        help="Minimum long-edge pixels before upsampling triggers (default: 1920).",
    )
    parser.add_argument(
        "--min-short-edge",
        type=int,
        default=1080,
        help="Minimum short-edge pixels before upsampling triggers (default: 1080).",
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=None,
        help="Optional path to a local .onnx super-resolution model.",
    )
    parser.add_argument(
        "--no-auto-download-onnx",
        action="store_true",
        help="Do not download a default ONNX model if none is present locally.",
    )
    parser.add_argument(
        "--allow-smart-zoom",
        action="store_true",
        help="Allow gentle zoom-in reframing (up to 15%) if it improves composition.",
    )
    parser.add_argument(
        "--orientation-mode",
        choices=["crop", "match_orientation", "blur_fill", "ai_outpaint"],
        default="crop",
        help=(
            "How to handle portrait <-> landscape orientation mismatches: "
            "crop, match_orientation, blur_fill, or ai_outpaint (default: crop)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process images even if they already exist in the corrected folder.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively process images inside subdirectories.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=94,
        help="Output JPEG/WebP quality 1-100 (default: 94).",
    )
    parser.add_argument(
        "--log-format",
        choices=["txt", "csv", "json", "all"],
        default="txt",
        help="Corrections log format saved in the output folder (default: txt).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Explicit path for saving the corrections log (.txt, .csv, or .json).",
    )
    parser.add_argument(
        "--no-save-log",
        action="store_true",
        help="Do not save a corrections log file in the output folder.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    source_dir: Path | None = args.source_dir
    if source_dir is None:
        source_dir = pick_folder_dialog()
        if source_dir is None:
            print("No folder selected. Exiting.")
            return 0

    try:
        ratio = AspectRatio.parse(args.ratio)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    config = ProcessConfig(
        source_dir=source_dir,
        output_dir=args.output,
        output_folder_name=args.output_name,
        crop=CropConfig(
            aspect_ratio=ratio,
            orientation_mode=args.orientation_mode,
            allow_smart_zoom=args.allow_smart_zoom,
        ),
        enhance=EnhanceConfig(
            enabled=not args.no_enhance,
            mode=args.enhancer,
            min_long_edge=args.min_long_edge,
            min_short_edge=args.min_short_edge,
            onnx_model_path=args.onnx_model,
            auto_download_onnx=not args.no_auto_download_onnx,
        ),
        jpeg_quality=args.quality,
        overwrite=args.overwrite,
        recursive=args.recursive,
        save_log=not args.no_save_log,
        log_format=args.log_format,
        log_file=args.log_file,
    )

    out_dir = config.resolved_output_dir()
    print(f"Source folder : {config.source_dir}")
    print(f"Output folder : {out_dir}")
    print(f"Aspect ratio  : {ratio.display_name}")
    print(
        f"Enhancement   : {'disabled' if args.no_enhance else args.enhancer} "
        f"(min {args.min_long_edge}x{args.min_short_edge})"
    )
    print("-" * 72)

    def _on_progress(idx: int, total: int, result: ImageResult) -> None:
        tag = result.status.upper().ljust(9)
        print(f"[{idx}/{total}] {tag} {result.source_path.name} — {result.message}")

    summary = process_folder(config, on_progress=_on_progress)
    print("-" * 72)
    print(
        f"Done. Found: {summary.total_found} | Processed: {summary.processed} "
        f"| Skipped: {summary.skipped} | Failed: {summary.failed}"
    )
    if summary.saved_log_paths:
        logs_str = ", ".join(str(p) for p in summary.saved_log_paths)
        print(f"Saved log(s)  : {logs_str}")
    return 1 if summary.failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
