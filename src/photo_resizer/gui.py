"""Desktop GUI for Photo Resizer built with native tkinter/ttk."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from photo_resizer.config import (
    ASPECT_RATIO_PRESETS,
    AspectRatio,
    CropConfig,
    EnhanceConfig,
    ProcessConfig,
)
from photo_resizer.correction_log import export_log_to_path
from photo_resizer.processor import BatchSummary, ImageResult, process_folder
from photo_resizer.settings_store import GuiSettings, load_settings, save_settings

ORIENTATION_MODE_LABELS: dict[str, str] = {
    "Crop to target ratio": "crop",
    "Match photo orientation (flip ratio)": "match_orientation",
    "Blurred backdrop fill (offline)": "blur_fill",
    "AI outpaint background (Gemini)": "ai_outpaint",
}
ORIENTATION_CODE_TO_LABEL: dict[str, str] = {v: k for k, v in ORIENTATION_MODE_LABELS.items()}

HELP_SECTIONS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "1. Folders",
        (
            (
                "Original Images Folder",
                "The folder containing your source photos (.jpg, .jpeg, .png, .webp, .bmp, .tif). "
                "Your original files are never modified.",
            ),
            (
                "Corrected Subfolder Name",
                "Name of the output folder created inside the source folder "
                "(default: 'corrected'). The output folder is automatically excluded "
                "from source scanning.",
            ),
            (
                "Custom Output Path",
                "Optional full path to save corrected images somewhere else instead of inside the "
                "source folder.",
            ),
        ),
    ),
    (
        "2. Target Aspect Ratio & Scene Framing",
        (
            (
                "Aspect Ratio Preset / Ratio (W:H)",
                "Choose a common preset (16:9, 3:2, 4:3, 5:4, 1:1, 4:5, 2:3, 9:16) or type any "
                "custom ratio like '21:9' or '1.5'. Photos already within 1% of the target ratio "
                "are kept without cropping.",
            ),
            (
                "How Scene Analysis Works (100% Offline)",
                "Uses local OpenCV + NumPy CV/ML (zero cloud AI): detects faces & headroom so "
                "heads aren't clipped, maps foreground subject saliency, and aligns detected "
                "horizons near the ideal rule-of-thirds line.",
            ),
            (
                "Skip images already in corrected folder",
                "When checked (default), re-running on the same folder skips any photo that "
                "already exists in the corrected output folder. Uncheck to re-process everything.",
            ),
            (
                "Include subfolders",
                "When checked, scans nested subdirectories inside the source folder and mirrors "
                "their folder structure in the output folder.",
            ),
            (
                "Allow smart zoom reframing (up to 15%)",
                "Unchecked (default) uses the largest possible crop rectangle (sliding along one "
                "axis only) to preserve maximum resolution. Checked also evaluates slightly "
                "tighter 85%-scale crops shifted in both X and Y, picking a zoomed-in crop only "
                "if it meaningfully improves composition.",
            ),
            (
                "Mismatched Orientation (Portrait <-> Landscape)",
                "Controls what happens when a portrait photo is processed with a landscape target "
                "ratio (e.g. 16:9), or vice versa:\n"
                "  • Crop to target ratio: Strictly crops every photo to the target ratio.\n"
                "  • Match photo orientation: Automatically flips the target ratio to match each "
                "photo (e.g. 16:9 becomes 9:16 for portraits, 3:2 becomes 2:3).\n"
                "  • Blurred backdrop fill (offline): Keeps the full photo intact inside the "
                "target frame over a softly blurred, color-matched backdrop (100% offline).\n"
                "  • AI outpaint background (Gemini): Uses Gemini (via GEMINI_API_KEY) to "
                "generate natural matching scenery on the empty sides, then feather-composites "
                "your untouched original photo back in so faces and details stay 100% faithful. "
                "Falls back to blurred backdrop fill automatically if offline or no key is set.",
            ),
        ),
    ),
    (
        "3. Low-Resolution Upsampling & Enhancement",
        (
            (
                "Enhance & upsample checkbox",
                "Automatically upsamples and sharpens any photo whose cropped dimensions fall "
                "below the minimum long or short edge thresholds (up to 4x).",
            ),
            (
                "Min Long Edge / Min Short Edge (px)",
                "Resolution floor (default 1920x1080) applied regardless of landscape or portrait "
                "orientation. Photos already above both thresholds are not upscaled.",
            ),
            (
                "Enhancer Tier — auto / offline",
                "100% local offline pipeline: uses fast OpenCV Lanczos + luminance sharpening for "
                "mild upscales (<1.4x) and local ONNX neural super-resolution for larger upscales "
                "(>=1.4x), falling back cleanly to OpenCV if no ONNX model is available.",
            ),
            (
                "Enhancer Tier — opencv",
                "Fastest, exact-fidelity classical upscaling (progressive Lanczos + bilateral "
                "denoise + LAB luminance sharpening). Never alters faces or fine textures.",
            ),
            (
                "Enhancer Tier — onnx",
                "Uses local ONNX super-resolution (e.g. Real-ESRGAN) for all upscaling. Place any "
                "NCHW .onnx super-res model in the project's models/ folder, browse to one below, "
                "or let it auto-download. Install GPU acceleration with: uv sync --extra gpu",
            ),
            (
                "Enhancer Tier — gemini",
                "Optional cloud AI restoration using Google Gemini API. Requires a Gemini API key "
                "(from Google AI Studio) pasted below or set in the GEMINI_API_KEY environment "
                "variable. Falls back to ONNX/OpenCV if unavailable.",
            ),
            (
                "Local ONNX Model / Gemini API Key",
                "Optional overrides. Leave ONNX blank to auto-detect from models/ or cache. "
                "Gemini API Key is only ever used when Enhancer Tier is set to 'gemini'.",
            ),
            (
                "Save Corrections Log / Save Log As...",
                "When checked, automatically saves a detailed per-image corrections log inside "
                "the output folder in your chosen format (txt, csv, json, or all) recording "
                "original/cropped/final sizes, exact crop box, change types, and tools used. "
                "Use 'Save Log As...' after a run to export the log to any folder.",
            ),
        ),
    ),
    (
        "4. Getting a Gemini API Key (Optional — Step-by-Step)",
        (
            (
                "Do you need a Gemini account?",
                "No. The default 'auto', 'offline', 'opencv', and 'onnx' tiers run 100% locally "
                "with no account or API key. You only need a Gemini API key if you want to use "
                "the optional 'gemini' cloud AI enhancement tier.",
            ),
            (
                "Step 1 — Create or sign in to a Google Account",
                "Gemini API keys are managed through Google AI Studio using a standard Google "
                "account (Gmail or Google Workspace). If you do not have one, go to "
                "https://accounts.google.com/signup and create a free Google account first.",
            ),
            (
                "Step 2 — Open Google AI Studio",
                "Visit https://aistudio.google.com in your browser and sign in with your Google "
                "account. On your first visit, review and accept the Google AI Studio / Gemini "
                "API Terms of Service to activate your developer access.",
            ),
            (
                "Step 3 — Create an API Key",
                "In Google AI Studio, click 'Get API key' in the left sidebar (or go directly to "
                "https://aistudio.google.com/apikey), then click 'Create API key'. Select an "
                "existing Google Cloud project or let AI Studio create a new default project "
                "automatically, then copy the generated key (starts with 'AIza...').",
            ),
            (
                "Step 4 — Check Billing / Quotas for Image Models",
                "Google AI Studio offers a free tier for basic testing, but image-output models "
                "(like gemini-2.5-flash-image) often have strict free limits or require linking "
                "a billing account. In AI Studio's 'API keys' or 'Billing' page, link a payment "
                "method if you see quota/billing errors—you only pay per image enhanced.",
            ),
            (
                "Step 5 — Enter the Key in Photo Resizer",
                "Keep your API key private (treat it like a password). Paste it into the "
                "'Gemini API Key' box in this window and set Enhancer Tier to 'gemini'. To avoid "
                "pasting it every time, save it in Windows PowerShell with:\n"
                "  setx GEMINI_API_KEY \"your-api-key-here\"\n"
                "and restart the app so it auto-fills automatically.",
            ),
        ),
    ),
)


class PhotoResizerApp:
    """Clean Windows-friendly desktop interface for batch photo cropping & upsampling."""

    def __init__(self, root: tk.Tk, settings_path: Path | None = None) -> None:
        self.root = root
        self.root.title("Photo Resizer — Smart Scene Crop & Upsample")
        self.root.minsize(720, 560)

        self._settings_path = settings_path
        saved = load_settings(self._settings_path)
        if saved.window_geometry:
            try:
                self.root.geometry(saved.window_geometry)
            except tk.TclError:
                self.root.geometry("820x660")
        else:
            self.root.geometry("820x660")

        self._worker: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._last_output_dir: Path | None = None
        self._last_summary: BatchSummary | None = None
        self._last_config: ProcessConfig | None = None
        self._help_window: tk.Toplevel | None = None

        # State variables initialized from last saved settings
        self.source_var = tk.StringVar(value=saved.source_dir)
        self.output_name_var = tk.StringVar(value=saved.output_folder_name)
        self.custom_output_var = tk.StringVar(value=saved.custom_output_dir)
        self.ratio_preset_var = tk.StringVar(value=saved.ratio_preset)
        self.custom_ratio_var = tk.StringVar(value=saved.custom_ratio)
        orient_label = ORIENTATION_CODE_TO_LABEL.get(
            saved.orientation_mode, "Crop to target ratio"
        )
        self.orientation_mode_var = tk.StringVar(value=orient_label)

        self.skip_existing_var = tk.BooleanVar(value=saved.skip_existing)
        self.recursive_var = tk.BooleanVar(value=saved.recursive)
        self.smart_zoom_var = tk.BooleanVar(value=saved.allow_smart_zoom)

        self.enhance_enabled_var = tk.BooleanVar(value=saved.enhance_enabled)
        self.enhancer_mode_var = tk.StringVar(value=saved.enhancer_mode)
        self.min_long_var = tk.IntVar(value=saved.min_long_edge)
        self.min_short_var = tk.IntVar(value=saved.min_short_edge)
        self.onnx_model_var = tk.StringVar(value=saved.onnx_model_path)
        self.gemini_key_var = tk.StringVar(value=os.environ.get("GEMINI_API_KEY", ""))
        self.save_log_var = tk.BooleanVar(value=saved.save_log)
        self.log_format_var = tk.StringVar(value=saved.log_format)

        self.status_var = tk.StringVar(value="Select an image folder to begin.")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._build_ui()
        self.root.bind("<F1>", lambda _event: self._show_help())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")

        main = ttk.Frame(self.root, padding=14)
        main.pack(fill=tk.BOTH, expand=True)

        # 1. Folder Selection
        folder_frame = ttk.LabelFrame(main, text="1. Folders", padding=10)
        folder_frame.pack(fill=tk.X, pady=(0, 10))
        folder_frame.columnconfigure(1, weight=1)

        ttk.Label(folder_frame, text="Original Images Folder:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 8), pady=4
        )
        ttk.Entry(folder_frame, textvariable=self.source_var).grid(
            row=0, column=1, sticky=tk.EW, pady=4
        )
        ttk.Button(folder_frame, text="Browse...", command=self._browse_source).grid(
            row=0, column=2, padx=(8, 0), pady=4
        )

        ttk.Label(folder_frame, text="Corrected Subfolder Name:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 8), pady=4
        )
        sub_row = ttk.Frame(folder_frame)
        sub_row.grid(row=1, column=1, columnspan=2, sticky=tk.EW, pady=4)
        sub_row.columnconfigure(2, weight=1)
        ttk.Entry(sub_row, textvariable=self.output_name_var, width=20).grid(
            row=0, column=0, sticky=tk.W
        )
        ttk.Label(sub_row, text="  or Custom Output Path:").grid(row=0, column=1, sticky=tk.W)
        ttk.Entry(sub_row, textvariable=self.custom_output_var).grid(
            row=0, column=2, sticky=tk.EW, padx=(6, 0)
        )
        ttk.Button(sub_row, text="Browse...", command=self._browse_output).grid(
            row=0, column=3, padx=(6, 0)
        )

        # 2. Aspect Ratio & Framing
        crop_frame = ttk.LabelFrame(main, text="2. Target Aspect Ratio & Scene Framing", padding=10)
        crop_frame.pack(fill=tk.X, pady=(0, 10))
        crop_frame.columnconfigure(3, weight=1)

        ttk.Label(crop_frame, text="Aspect Ratio Preset:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 8), pady=4
        )
        preset_values = list(ASPECT_RATIO_PRESETS.keys()) + ["Custom"]
        preset_combo = ttk.Combobox(
            crop_frame,
            textvariable=self.ratio_preset_var,
            values=preset_values,
            state="readonly",
            width=24,
        )
        preset_combo.grid(row=0, column=1, sticky=tk.W, pady=4)
        preset_combo.bind("<<ComboboxSelected>>", self._on_preset_change)

        ttk.Label(crop_frame, text="Ratio (W:H):").grid(
            row=0, column=2, sticky=tk.W, padx=(16, 6), pady=4
        )
        self.custom_ratio_entry = ttk.Entry(
            crop_frame, textvariable=self.custom_ratio_var, width=12
        )
        self.custom_ratio_entry.grid(row=0, column=3, sticky=tk.W, pady=4)

        ttk.Label(crop_frame, text="Mismatched Orientation:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 8), pady=4
        )
        ttk.Combobox(
            crop_frame,
            textvariable=self.orientation_mode_var,
            values=list(ORIENTATION_MODE_LABELS.keys()),
            state="readonly",
            width=34,
        ).grid(row=1, column=1, columnspan=3, sticky=tk.W, pady=4)

        opts_row = ttk.Frame(crop_frame)
        opts_row.grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(6, 0))
        ttk.Checkbutton(
            opts_row,
            text="Skip images already in corrected folder",
            variable=self.skip_existing_var,
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(
            opts_row,
            text="Include subfolders",
            variable=self.recursive_var,
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(
            opts_row,
            text="Allow smart zoom reframing (up to 15%)",
            variable=self.smart_zoom_var,
        ).pack(side=tk.LEFT)

        # 3. Upsampling & Enhancement
        enh_frame = ttk.LabelFrame(
            main, text="3. Low-Resolution Upsampling & Enhancement", padding=10
        )
        enh_frame.pack(fill=tk.X, pady=(0, 10))
        enh_frame.columnconfigure(3, weight=1)

        ttk.Checkbutton(
            enh_frame,
            text="Enhance & upsample cropped images below minimum resolution",
            variable=self.enhance_enabled_var,
        ).grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 6))

        ttk.Label(enh_frame, text="Min Long Edge (px):").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 6), pady=4
        )
        ttk.Spinbox(
            enh_frame, from_=320, to=8192, increment=64, textvariable=self.min_long_var, width=10
        ).grid(row=1, column=1, sticky=tk.W, pady=4)

        ttk.Label(enh_frame, text="Min Short Edge (px):").grid(
            row=1, column=2, sticky=tk.W, padx=(16, 6), pady=4
        )
        ttk.Spinbox(
            enh_frame, from_=240, to=8192, increment=64, textvariable=self.min_short_var, width=10
        ).grid(row=1, column=3, sticky=tk.W, pady=4)

        ttk.Label(enh_frame, text="Enhancer Tier:").grid(
            row=2, column=0, sticky=tk.W, padx=(0, 6), pady=4
        )
        mode_combo = ttk.Combobox(
            enh_frame,
            textvariable=self.enhancer_mode_var,
            values=["auto", "offline", "opencv", "onnx", "gemini"],
            state="readonly",
            width=14,
        )
        mode_combo.grid(row=2, column=1, sticky=tk.W, pady=4)
        ttk.Label(
            enh_frame,
            text="auto/offline: OpenCV <1.4x, local ONNX >=1.4x (zero cloud AI unless 'gemini')",
        ).grid(row=2, column=2, columnspan=2, sticky=tk.W, padx=(12, 0), pady=4)

        ttk.Label(enh_frame, text="Local ONNX Model (optional):").grid(
            row=3, column=0, sticky=tk.W, padx=(0, 6), pady=4
        )
        ttk.Entry(enh_frame, textvariable=self.onnx_model_var).grid(
            row=3, column=1, columnspan=2, sticky=tk.EW, pady=4
        )
        ttk.Button(enh_frame, text="Browse .onnx", command=self._browse_onnx).grid(
            row=3, column=3, sticky=tk.W, padx=(8, 0), pady=4
        )

        ttk.Label(enh_frame, text="Gemini API Key (optional):").grid(
            row=4, column=0, sticky=tk.W, padx=(0, 6), pady=4
        )
        ttk.Entry(enh_frame, textvariable=self.gemini_key_var, show="*").grid(
            row=4, column=1, columnspan=2, sticky=tk.EW, pady=4
        )
        ttk.Label(enh_frame, text="(Used for Gemini tier or AI outpaint)").grid(
            row=4, column=3, sticky=tk.W, padx=(8, 0), pady=4
        )

        log_opt_row = ttk.Frame(enh_frame)
        log_opt_row.grid(row=5, column=0, columnspan=4, sticky=tk.W, pady=(6, 0))
        ttk.Checkbutton(
            log_opt_row,
            text="Save detailed corrections log in output folder",
            variable=self.save_log_var,
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(log_opt_row, text="Format:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(
            log_opt_row,
            textvariable=self.log_format_var,
            values=["txt", "csv", "json", "all"],
            state="readonly",
            width=8,
        ).pack(side=tk.LEFT)

        # 4. Controls & Progress
        ctrl_row = ttk.Frame(main)
        ctrl_row.pack(fill=tk.X, pady=(0, 8))

        self.run_btn = ttk.Button(
            ctrl_row, text="Process Images", command=self._start_processing
        )
        self.run_btn.pack(side=tk.LEFT)

        self.cancel_btn = ttk.Button(
            ctrl_row, text="Cancel", command=self._cancel_processing, state=tk.DISABLED
        )
        self.cancel_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.help_btn = ttk.Button(
            ctrl_row,
            text="Help (?)",
            command=self._show_help,
        )
        self.help_btn.pack(side=tk.RIGHT, padx=(8, 0))

        self.save_log_btn = ttk.Button(
            ctrl_row,
            text="Save Log As...",
            command=self._save_log_as,
            state=tk.DISABLED,
        )
        self.save_log_btn.pack(side=tk.RIGHT, padx=(8, 0))

        self.open_out_btn = ttk.Button(
            ctrl_row,
            text="Open Corrected Folder",
            command=self._open_output_folder,
            state=tk.DISABLED,
        )
        self.open_out_btn.pack(side=tk.RIGHT)

        self.progress_bar = ttk.Progressbar(
            main, variable=self.progress_var, maximum=100.0, mode="determinate"
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(main, textvariable=self.status_var).pack(anchor=tk.W, pady=(0, 6))

        # 5. Activity Log
        log_frame = ttk.LabelFrame(main, text="Processing Log", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, height=11, font=("Consolas", 9), state=tk.DISABLED
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _on_preset_change(self, _event: object = None) -> None:
        key = self.ratio_preset_var.get()
        if key in ASPECT_RATIO_PRESETS:
            self.custom_ratio_var.set(ASPECT_RATIO_PRESETS[key].display_name)

    def _browse_source(self) -> None:
        initial = self.source_var.get().strip()
        kwargs: dict[str, str] = {"title": "Select Original Image Folder"}
        if initial and Path(initial).is_dir():
            kwargs["initialdir"] = initial
        selected = filedialog.askdirectory(**kwargs)
        if selected:
            self.source_var.set(selected)
            self._save_current_settings()

    def _browse_output(self) -> None:
        initial = self.custom_output_var.get().strip() or self.source_var.get().strip()
        kwargs: dict[str, str] = {"title": "Select Corrected Images Output Folder"}
        if initial and Path(initial).is_dir():
            kwargs["initialdir"] = initial
        selected = filedialog.askdirectory(**kwargs)
        if selected:
            self.custom_output_var.set(selected)
            self._save_current_settings()

    def _browse_onnx(self) -> None:
        initial = self.onnx_model_var.get().strip()
        initial_dir = str(Path(initial).parent) if initial and Path(initial).parent.is_dir() else ""
        kwargs: dict[str, object] = {
            "title": "Select Local ONNX Super-Resolution Model",
            "filetypes": [("ONNX models", "*.onnx"), ("All files", "*.*")],
        }
        if initial_dir:
            kwargs["initialdir"] = initial_dir
        selected = filedialog.askopenfilename(**kwargs)  # type: ignore[arg-type]
        if selected:
            self.onnx_model_var.set(selected)
            self._save_current_settings()

    def _collect_settings(self) -> GuiSettings:
        try:
            min_long = int(self.min_long_var.get())
        except (tk.TclError, ValueError):
            min_long = 1920
        try:
            min_short = int(self.min_short_var.get())
        except (tk.TclError, ValueError):
            min_short = 1080

        return GuiSettings(
            source_dir=self.source_var.get().strip(),
            output_folder_name=self.output_name_var.get().strip() or "corrected",
            custom_output_dir=self.custom_output_var.get().strip(),
            ratio_preset=self.ratio_preset_var.get(),
            custom_ratio=self.custom_ratio_var.get().strip() or "16:9",
            orientation_mode=ORIENTATION_MODE_LABELS.get(
                self.orientation_mode_var.get(), "crop"
            ),
            skip_existing=self.skip_existing_var.get(),
            recursive=self.recursive_var.get(),
            allow_smart_zoom=self.smart_zoom_var.get(),
            enhance_enabled=self.enhance_enabled_var.get(),
            enhancer_mode=self.enhancer_mode_var.get(),
            min_long_edge=min_long,
            min_short_edge=min_short,
            onnx_model_path=self.onnx_model_var.get().strip(),
            save_log=self.save_log_var.get(),
            log_format=self.log_format_var.get(),
            window_geometry=self.root.winfo_geometry(),
        )

    def _save_current_settings(self) -> None:
        try:
            save_settings(self._collect_settings(), self._settings_path)
        except Exception:
            pass

    def _on_close(self) -> None:
        self._save_current_settings()
        self.root.destroy()

    def _append_log(self, line: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, line + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _build_config(self) -> ProcessConfig | None:
        src_text = self.source_var.get().strip()
        if not src_text:
            self._browse_source()
            src_text = self.source_var.get().strip()
            if not src_text:
                return None

        source_dir = Path(src_text)
        if not source_dir.is_dir():
            messagebox.showerror("Invalid Folder", f"Source folder does not exist:\n{source_dir}")
            return None

        try:
            ratio = AspectRatio.parse(self.custom_ratio_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid Aspect Ratio", str(exc))
            return None

        out_custom = self.custom_output_var.get().strip()
        output_dir = Path(out_custom) if out_custom else None
        out_name = self.output_name_var.get().strip() or "corrected"

        onnx_text = self.onnx_model_var.get().strip()
        onnx_path = Path(onnx_text) if onnx_text else None
        gemini_key = self.gemini_key_var.get().strip() or None

        return ProcessConfig(
            source_dir=source_dir,
            output_dir=output_dir,
            output_folder_name=out_name,
            crop=CropConfig(
                aspect_ratio=ratio,
                orientation_mode=ORIENTATION_MODE_LABELS.get(
                    self.orientation_mode_var.get(), "crop"
                ),  # type: ignore[arg-type]
                allow_smart_zoom=self.smart_zoom_var.get(),
            ),
            enhance=EnhanceConfig(
                enabled=self.enhance_enabled_var.get(),
                mode=self.enhancer_mode_var.get(),  # type: ignore[arg-type]
                min_long_edge=int(self.min_long_var.get()),
                min_short_edge=int(self.min_short_var.get()),
                onnx_model_path=onnx_path,
                gemini_api_key=gemini_key,
            ),
            overwrite=not self.skip_existing_var.get(),
            recursive=self.recursive_var.get(),
            save_log=self.save_log_var.get(),
            log_format=self.log_format_var.get(),  # type: ignore[arg-type]
        )

    def _start_processing(self) -> None:
        config = self._build_config()
        if config is None:
            return

        self._save_current_settings()
        self._cancel_event.clear()
        self._last_output_dir = config.resolved_output_dir()
        self._last_config = config
        self._last_summary = None
        self._clear_log()
        self.progress_var.set(0.0)
        self.status_var.set("Scanning folder...")
        self.run_btn.configure(state=tk.DISABLED)
        self.cancel_btn.configure(state=tk.NORMAL)
        self.open_out_btn.configure(state=tk.DISABLED)
        self.save_log_btn.configure(state=tk.DISABLED)

        self._append_log(f"Source : {config.source_dir}")
        self._append_log(f"Output : {self._last_output_dir}")
        self._append_log(
            f"Target : {config.crop.aspect_ratio.display_name} | "
            f"Enhancer: {config.enhance.mode if config.enhance.enabled else 'disabled'}"
        )
        self._append_log("-" * 70)

        self._worker = threading.Thread(target=self._run_worker, args=(config,), daemon=True)
        self._worker.start()

    def _run_worker(self, config: ProcessConfig) -> None:
        def on_progress(idx: int, total: int, result: ImageResult) -> None:
            pct = (idx / max(1, total)) * 100.0
            tag = result.status.upper().ljust(9)
            line = f"[{idx}/{total}] {tag} {result.source_path.name} — {result.message}"
            self.root.after(0, self._update_progress, pct, idx, total, line)

        try:
            summary = process_folder(
                config,
                on_progress=on_progress,
                cancel_event=self._cancel_event,
            )
            self.root.after(0, self._on_finished, summary)
        except Exception as exc:
            self.root.after(0, self._on_error, str(exc))

    def _update_progress(self, pct: float, idx: int, total: int, line: str) -> None:
        self.progress_var.set(pct)
        self.status_var.set(f"Processing {idx} of {total} images...")
        self._append_log(line)

    def _on_finished(self, summary: BatchSummary) -> None:
        self.run_btn.configure(state=tk.NORMAL)
        self.cancel_btn.configure(state=tk.DISABLED)
        self._last_summary = summary
        if summary.results:
            self.save_log_btn.configure(state=tk.NORMAL)
        if summary.output_dir.exists():
            self.open_out_btn.configure(state=tk.NORMAL)
            self._last_output_dir = summary.output_dir

        state_word = "Cancelled" if summary.cancelled else "Completed"
        msg = (
            f"{state_word}: {summary.processed} processed, "
            f"{summary.skipped} skipped, {summary.failed} failed "
            f"(of {summary.total_found} found)."
        )
        self.status_var.set(msg)
        self._append_log("-" * 70)
        self._append_log(msg)
        if summary.saved_log_paths:
            self._append_log(
                "Saved log(s): " + ", ".join(str(p) for p in summary.saved_log_paths)
            )
        if not summary.cancelled:
            self.progress_var.set(100.0)

    def _on_error(self, error_message: str) -> None:
        self.run_btn.configure(state=tk.NORMAL)
        self.cancel_btn.configure(state=tk.DISABLED)
        self.status_var.set(f"Error: {error_message}")
        self._append_log(f"ERROR: {error_message}")
        messagebox.showerror("Processing Error", error_message)

    def _cancel_processing(self) -> None:
        self._cancel_event.set()
        self.status_var.set("Cancelling after current image...")
        self.cancel_btn.configure(state=tk.DISABLED)

    def _save_log_as(self) -> None:
        if self._last_summary is None or self._last_config is None:
            return
        initial_dir = (
            str(self._last_output_dir)
            if self._last_output_dir and self._last_output_dir.is_dir()
            else ""
        )
        selected = filedialog.asksaveasfilename(
            title="Save Corrections Log As",
            initialdir=initial_dir or None,
            initialfile="corrections_log.txt",
            defaultextension=".txt",
            filetypes=[
                ("Text Log (*.txt)", "*.txt"),
                ("CSV Spreadsheet (*.csv)", "*.csv"),
                ("JSON Report (*.json)", "*.json"),
            ],
        )
        if not selected:
            return
        try:
            saved_path = export_log_to_path(
                self._last_summary, self._last_config, Path(selected), append=False
            )
            self._append_log(f"Exported log to: {saved_path}")
            self.status_var.set(f"Log saved to {saved_path.name}")
        except Exception as exc:
            messagebox.showerror("Save Log Error", f"Failed to save log:\n{exc}")

    def _open_output_folder(self) -> None:
        if self._last_output_dir is None or not self._last_output_dir.exists():
            return
        if sys.platform == "win32":
            os.startfile(str(self._last_output_dir))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(self._last_output_dir)])
        else:
            subprocess.Popen(["xdg-open", str(self._last_output_dir)])

    def _show_help(self) -> None:
        """Open a scrollable reference dialog explaining all app options."""
        if self._help_window is not None and self._help_window.winfo_exists():
            self._help_window.lift()
            self._help_window.focus_force()
            return

        win = tk.Toplevel(self.root)
        win.title("Photo Resizer — Options Help")
        win.geometry("680x580")
        win.minsize(520, 420)
        win.transient(self.root)
        self._help_window = win

        container = ttk.Frame(win, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        text = scrolledtext.ScrolledText(
            container,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            padx=10,
            pady=8,
            relief=tk.FLAT,
        )
        text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        text.tag_configure("section", font=("Segoe UI", 11, "bold"), spacing1=10, spacing3=4)
        text.tag_configure("option", font=("Segoe UI", 10, "bold"), spacing1=6)
        text.tag_configure("body", font=("Segoe UI", 10), lmargin1=14, lmargin2=14, spacing3=4)

        for section_title, items in HELP_SECTIONS:
            text.insert(tk.END, f"{section_title}\n", "section")
            for opt_name, desc in items:
                text.insert(tk.END, f"• {opt_name}\n", "option")
                text.insert(tk.END, f"{desc}\n", "body")

        text.configure(state=tk.DISABLED)

        btn_row = ttk.Frame(container)
        btn_row.pack(fill=tk.X)
        ttk.Label(btn_row, text="Tip: Press F1 anytime to open this guide.").pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Close", command=win.destroy).pack(side=tk.RIGHT)
        win.bind("<Escape>", lambda _event: win.destroy())


def main() -> int:
    root = tk.Tk()
    PhotoResizerApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
