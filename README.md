# Photo Resizer

Offline-first Python desktop app and CLI that batch-processes a folder of images, intelligently cropping them to a target aspect ratio using local CV/ML scene analysis, and automatically enhancing/upsampling low-resolution crops.

## Features

- **100% Offline Scene Analysis & Smart Cropping** (OpenCV + NumPy, zero cloud AI calls):
  - **Face & Headroom Protection**: Detects frontal and profile faces via bundled OpenCV Haar cascades and protects head/shoulder margins so subjects are never awkwardly sliced.
  - **Multi-Cue Subject Saliency**: Combines Spectral Residual FFT saliency, CIE-Lab border contrast, and local edge sharpness to keep in-focus subjects intact.
  - **Horizon Line Placement**: Detects dominant horizons via Hough line voting + vertical Lab color gradients and aligns them near the ideal rule-of-thirds line (favoring upper or lower third depending on whether sky or foreground is more interesting).
  - **EXIF Safe**: Auto-orients rotated camera/phone photos before analysis and resets the Orientation tag on save so viewers never double-rotate.
- **Tiered Low-Resolution Enhancement & Upsampling**:
  - **Mild (`< 1.4x`)**: Fast offline OpenCV progressive Lanczos + edge-preserving denoise + LAB luminance sharpening.
  - **Moderate/Heavy (`>= 1.4x`)**: Local offline **ONNX Super-Resolution** (`onnxruntime`, tiled overlap blending, auto-selects Windows DirectML/CUDA GPU if available, automatic fallback to OpenCV if offline/no model).
  - **Optional Gemini AI Mode**: Available when explicitly selected (`--enhancer gemini` or GUI tier = `gemini`) with `GEMINI_API_KEY` set.
- **Idempotent Folder Processing**:
  - Creates a dedicated corrected output folder (default `<selected_folder>\corrected`).
  - Automatically skips any images already present in the corrected folder on re-runs (unless `--overwrite` / uncheck skip is enabled).
  - Writes a `processing_report.json` explaining every crop and upsampling decision.

## Setup

Using [`uv`](https://docs.astral.sh/uv/):

```powershell
cd C:\Users\john\Documents\projects\photo_resizer
uv sync --group dev
```

Optional extras:
- Windows GPU acceleration for local ONNX super-resolution: `uv sync --extra gpu`
- Official Gemini SDK (optional; REST fallback via `httpx` also works out of the box): `uv sync --extra ai`

## Usage

### Desktop GUI (with Native Folder Picker)
Double-click `Launch-PhotoResizer.bat`, or run:

```powershell
uv run photo-resizer
```

### CLI
Run with a folder path (or omit the path to pop up the native Windows folder picker):

```powershell
uv run photo-resizer-cli "C:\Users\john\Pictures\Trip" --ratio 16:9
uv run photo-resizer-cli --ratio 4:5 --enhancer offline
```

## Development & Tests

```powershell
uv run ruff check src tests
uv run pytest -v
```
