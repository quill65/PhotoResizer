"""Tiered upsampling and enhancement pipeline for low-resolution crops.

Tiers (using as little cloud AI as possible):
1. OpenCV (fast, exact fidelity): mild upscaling (<1.4x) or offline fallback
2. Local ONNX ML (Real-ESRGAN / SwinIR NCHW super-resolution with tiled overlap blending)
3. Optional Gemini AI (generative restoration only when explicitly enabled + API key set)
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import httpx
import numpy as np
import onnxruntime as ort
from PIL import Image

from photo_resizer.config import EnhanceConfig

DEFAULT_ONNX_MODEL_NAME = "realesr-general-x4v3.onnx"
DEFAULT_ONNX_MODEL_URLS: tuple[str, ...] = (
    "https://huggingface.co/qualcomm/Real-ESRGAN-General-x4v3/resolve/main/Real-ESRGAN-General-x4v3.onnx",
    "https://huggingface.co/spaces/Xenova/real-esrgan-web/resolve/main/models/realesr-general-x4v3.onnx",
)


class ModelUnavailableError(RuntimeError):
    """Raised when an optional enhancement backend or model cannot be loaded."""


@dataclass(frozen=True)
class EnhanceResult:
    """Result returned by the enhancement pipeline."""

    image: Image.Image
    was_enhanced: bool
    backend: str
    scale_factor: float
    note: str


class ImageEnhancer(Protocol):
    name: str

    def enhance(self, image: Image.Image, target_w: int, target_h: int) -> Image.Image: ...


def compute_upscale_target(
    width: int,
    height: int,
    config: EnhanceConfig,
    target_ratio: float | None = None,
) -> tuple[int, int, float]:
    """Return (target_w, target_h, scale_factor) if image is below minimum resolution."""
    if not config.enabled or width <= 0 or height <= 0:
        return width, height, 1.0

    long_edge = max(width, height)
    short_edge = min(width, height)

    scale_long = config.min_long_edge / long_edge if config.min_long_edge > 0 else 1.0
    scale_short = config.min_short_edge / short_edge if config.min_short_edge > 0 else 1.0
    needed = max(1.0, scale_long, scale_short)

    if needed <= 1.02:
        return width, height, 1.0

    scale = min(needed, max(1.0, config.max_upscale_factor))

    # When an explicit target aspect ratio is provided, snap dimensions cleanly to it
    # (avoids +/-1px rounding drift like 1281x720 instead of exact 1280x720).
    if target_ratio is not None and target_ratio > 0:
        max_factor = max(1.0, config.max_upscale_factor)
        if target_ratio >= 1.0:
            desired_w = max(
                float(width),
                float(config.min_long_edge),
                float(config.min_short_edge) * target_ratio,
            )
            cand_w = max(width, int(round(min(desired_w, width * max_factor))))
            cand_h = max(height, int(round(cand_w / target_ratio)))
        else:
            desired_h = max(
                float(height),
                float(config.min_long_edge),
                float(config.min_short_edge) / target_ratio,
            )
            cand_h = max(height, int(round(min(desired_h, height * max_factor))))
            cand_w = max(width, int(round(cand_h * target_ratio)))
        eff_scale = max(cand_w / width, cand_h / height)
        return cand_w, cand_h, eff_scale

    target_w = max(width, int(round(width * scale)))
    target_h = max(height, int(round(height * scale)))
    return target_w, target_h, scale


class OpenCVEnhancer:
    """Fast, exact-fidelity offline upscaler using progressive Lanczos + luminance sharpening."""

    name = "opencv"

    def enhance(self, image: Image.Image, target_w: int, target_h: int) -> Image.Image:
        has_alpha = image.mode == "RGBA"
        alpha_channel: Image.Image | None = None
        if has_alpha:
            alpha_channel = image.getchannel("A").resize(
                (target_w, target_h), Image.Resampling.LANCZOS
            )

        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        cur_h, cur_w = rgb.shape[:2]

        # Subtle edge-preserving denoise before upscaling so compression artifacts aren't amplified
        denoised = cv2.bilateralFilter(rgb, d=5, sigmaColor=18, sigmaSpace=18)
        work = cv2.addWeighted(rgb, 0.65, denoised, 0.35, 0)

        # Progressive 2x Lanczos steps reduce ringing for larger scale factors
        while cur_w * 2 < target_w and cur_h * 2 < target_h:
            cur_w *= 2
            cur_h *= 2
            work = cv2.resize(work, (cur_w, cur_h), interpolation=cv2.INTER_LANCZOS4)

        upscaled = cv2.resize(work, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        # Enhance micro-contrast and sharpness strictly in LAB luminance (avoids color noise)
        lab = cv2.cvtColor(upscaled, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_chan)
        l_mix = cv2.addWeighted(l_chan, 0.78, l_clahe, 0.22, 0)

        blurred = cv2.GaussianBlur(l_mix, (0, 0), sigmaX=1.2)
        l_sharp = cv2.addWeighted(l_mix, 1.38, blurred, -0.38, 0)

        merged = cv2.merge((l_sharp, a_chan, b_chan))
        out_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        result = Image.fromarray(out_rgb, mode="RGB")

        if alpha_channel is not None:
            result = result.convert("RGBA")
            result.putalpha(alpha_channel)
        return result


def _default_model_cache_dir() -> Path:
    base = os.environ.get("PHOTO_RESIZER_CACHE_DIR")
    if base:
        return Path(base) / "models"
    return Path.home() / ".cache" / "photo_resizer" / "models"


def find_local_onnx_model(explicit_path: Path | None = None) -> Path | None:
    """Locate a user-provided or cached .onnx super-resolution model."""
    if explicit_path is not None:
        return explicit_path if explicit_path.is_file() else None

    # Check project-local models/ directory first, then user cache
    candidates_dirs = [
        Path.cwd() / "models",
        Path(__file__).resolve().parents[2] / "models",
        _default_model_cache_dir(),
    ]
    for directory in candidates_dirs:
        if not directory.is_dir():
            continue
        preferred = directory / DEFAULT_ONNX_MODEL_NAME
        if preferred.is_file():
            return preferred
        onnx_files = sorted(directory.glob("*.onnx"))
        if onnx_files:
            return onnx_files[0]
    return None


def download_default_onnx_model(target_dir: Path | None = None, timeout: float = 45.0) -> Path:
    """Download a lightweight open-source Real-ESRGAN ONNX model into local cache."""
    dest_dir = target_dir or _default_model_cache_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / DEFAULT_ONNX_MODEL_NAME
    if dest_path.is_file() and dest_path.stat().st_size > 100_000:
        return dest_path

    last_error: Exception | None = None
    for url in DEFAULT_ONNX_MODEL_URLS:
        tmp_path: Path | None = None
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as resp:
                resp.raise_for_status()
                with tempfile.NamedTemporaryFile(
                    dir=dest_dir, prefix="dl_", suffix=".onnx.tmp", delete=False
                ) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    for chunk in resp.iter_bytes(chunk_size=1024 * 256):
                        tmp_file.write(chunk)
            if tmp_path is not None and tmp_path.stat().st_size > 100_000:
                tmp_path.replace(dest_path)
                return dest_path
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            last_error = exc
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    raise ModelUnavailableError(
        f"Could not download default ONNX super-resolution model ({last_error})"
    )


class OnnxSuperResEnhancer:
    """Local offline neural super-resolution via ONNX Runtime with tiled overlap blending.

    Compatible with standard NCHW float32 RGB [0, 1] models (Real-ESRGAN, SwinIR, 4x-UltraSharp).
    Automatically uses Windows DirectML or CUDA when available, otherwise CPU.
    """

    name = "onnx"

    def __init__(
        self,
        model_path: Path | None = None,
        auto_download: bool = True,
        tile_size: int = 512,
        tile_overlap: int = 32,
    ) -> None:
        self._explicit_path = model_path
        self._auto_download = auto_download
        self._tile_size = max(64, tile_size)
        self._tile_overlap = max(8, tile_overlap)
        self._session: ort.InferenceSession | None = None
        self._input_name: str = ""
        self._output_name: str = ""
        self._model_scale: int = 4
        self._loaded_model_path: Path | None = None
        self._active_provider: str = "CPUExecutionProvider"

    @property
    def active_provider(self) -> str:
        return self._active_provider

    @property
    def loaded_model_path(self) -> Path | None:
        return self._loaded_model_path

    def _ensure_session(self) -> ort.InferenceSession:
        if self._session is not None:
            return self._session

        model_path = find_local_onnx_model(self._explicit_path)
        if model_path is None:
            if self._explicit_path is not None:
                raise ModelUnavailableError(f"ONNX model not found at {self._explicit_path}")
            if not self._auto_download:
                raise ModelUnavailableError(
                    "No local .onnx super-resolution model found in models/ or cache"
                )
            model_path = download_default_onnx_model()

        available = ort.get_available_providers()
        preferred = ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred if p in available] or ["CPUExecutionProvider"]

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            session = ort.InferenceSession(
                str(model_path), sess_options=sess_opts, providers=providers
            )
        except Exception as exc:
            msg = f"Failed to load ONNX model {model_path.name}: {exc}"
            raise ModelUnavailableError(msg) from exc

        self._session = session
        self._loaded_model_path = model_path
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name
        used = session.get_providers()
        self._active_provider = used[0] if used else "CPUExecutionProvider"

        # Probe native integer upscale factor using a tiny dummy patch
        self._model_scale = self._probe_scale(session)
        return session

    def _probe_scale(self, session: ort.InferenceSession) -> int:
        inp = session.get_inputs()[0]
        shape = inp.shape
        # Handle fixed spatial input models gracefully if any
        probe_h = shape[2] if len(shape) >= 4 and isinstance(shape[2], int) and shape[2] > 0 else 16
        probe_w = shape[3] if len(shape) >= 4 and isinstance(shape[3], int) and shape[3] > 0 else 16
        dummy = np.zeros((1, 3, probe_h, probe_w), dtype=np.float32)
        out = session.run([self._output_name], {self._input_name: dummy})[0]
        if out.ndim == 4 and out.shape[2] >= probe_h:
            return max(1, int(round(out.shape[2] / probe_h)))
        return 4

    def _run_tile(self, session: ort.InferenceSession, tile_rgb: np.ndarray) -> np.ndarray:
        """Run single RGB uint8 tile through NCHW float32 [0,1] ONNX model."""
        arr = tile_rgb.astype(np.float32) / 255.0
        nchw = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
        out = session.run([self._output_name], {self._input_name: nchw})[0]
        out_hwc = np.transpose(out[0], (1, 2, 0))
        return np.clip(out_hwc, 0.0, 1.0)

    def _upscale_tiled(self, session: ort.InferenceSession, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        scale = self._model_scale
        tile = self._tile_size
        overlap = min(self._tile_overlap, tile // 4)

        if h <= tile and w <= tile:
            out_f = self._run_tile(session, rgb)
            return np.clip(np.round(out_f * 255.0), 0, 255).astype(np.uint8)

        out_h, out_w = h * scale, w * scale
        accum = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weight = np.zeros((out_h, out_w, 1), dtype=np.float32)

        step = max(1, tile - overlap)
        ys = list(range(0, max(1, h - tile + 1), step))
        if ys[-1] + tile < h:
            ys.append(max(0, h - tile))
        xs = list(range(0, max(1, w - tile + 1), step))
        if xs[-1] + tile < w:
            xs.append(max(0, w - tile))

        for y in ys:
            for x in xs:
                y2 = min(h, y + tile)
                x2 = min(w, x + tile)
                patch = rgb[y:y2, x:x2]
                pred = self._run_tile(session, patch)

                ph, pw = pred.shape[:2]
                wy = np.ones(ph, dtype=np.float32)
                wx = np.ones(pw, dtype=np.float32)
                feather = min(overlap * scale, ph // 2, pw // 2)
                if feather > 1:
                    ramp = np.linspace(0.05, 1.0, feather, dtype=np.float32)
                    if y > 0:
                        wy[:feather] = ramp
                    if y2 < h:
                        wy[-feather:] = ramp[::-1]
                    if x > 0:
                        wx[:feather] = ramp
                    if x2 < w:
                        wx[-feather:] = ramp[::-1]
                w2d = (wy[:, None] * wx[None, :])[:, :, None]

                oy1, ox1 = y * scale, x * scale
                oy2, ox2 = oy1 + ph, ox1 + pw
                accum[oy1:oy2, ox1:ox2] += pred * w2d
                weight[oy1:oy2, ox1:ox2] += w2d

        blended = accum / np.maximum(weight, 1e-6)
        return np.clip(np.round(blended * 255.0), 0, 255).astype(np.uint8)

    def enhance(self, image: Image.Image, target_w: int, target_h: int) -> Image.Image:
        session = self._ensure_session()
        has_alpha = image.mode == "RGBA"
        alpha_channel: Image.Image | None = None
        if has_alpha:
            alpha_channel = image.getchannel("A").resize(
                (target_w, target_h), Image.Resampling.LANCZOS
            )

        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        upscaled = self._upscale_tiled(session, rgb)

        if upscaled.shape[1] != target_w or upscaled.shape[0] != target_h:
            interp = (
                cv2.INTER_AREA
                if upscaled.shape[1] > target_w and upscaled.shape[0] > target_h
                else cv2.INTER_LANCZOS4
            )
            upscaled = cv2.resize(upscaled, (target_w, target_h), interpolation=interp)

        result = Image.fromarray(upscaled, mode="RGB")
        if alpha_channel is not None:
            result = result.convert("RGBA")
            result.putalpha(alpha_channel)
        return result


class GeminiEnhancer:
    """Optional cloud AI photographic enhancer via Gemini API.

    Only called when explicitly requested or opted into AND GEMINI_API_KEY is available.
    Uses the `google-genai` SDK if installed, otherwise calls the REST API via `httpx`.
    """

    name = "gemini"

    def __init__(self, api_key: str | None, model: str = "gemini-2.5-flash-image") -> None:
        self._api_key = api_key
        self._model = model

    def enhance(self, image: Image.Image, target_w: int, target_h: int) -> Image.Image:
        if not self._api_key:
            raise ModelUnavailableError("GEMINI_API_KEY is not configured")

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        png_bytes = buf.getvalue()

        prompt = (
            "Restore and enhance the sharpness, clarity, and natural photographic texture "
            "of this cropped photo for high-resolution output. Preserve the exact composition, "
            "aspect ratio, lighting, colors, faces, and every object faithfully—do not add, "
            "remove, or alter subjects."
        )

        enhanced_bytes = self._call_gemini_rest(png_bytes, prompt)
        with Image.open(io.BytesIO(enhanced_bytes)) as returned:
            out = returned.convert("RGBA" if image.mode == "RGBA" else "RGB")

        if out.size != (target_w, target_h):
            out = out.resize((target_w, target_h), Image.Resampling.LANCZOS)
        return out

    def _call_gemini_rest(self, png_bytes: bytes, prompt: str) -> bytes:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent"
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": base64.b64encode(png_bytes).decode("ascii"),
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {"responseModalities": ["IMAGE"]},
        }

        try:
            resp = httpx.post(
                url,
                headers={
                    "x-goog-api-key": self._api_key or "",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=90.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise ModelUnavailableError(f"Gemini API request failed: {exc}") from exc

        for candidate in data.get("candidates", []):
            parts = candidate.get("content", {}).get("parts", [])
            for part in parts:
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    return base64.b64decode(inline["data"])

        raise ModelUnavailableError("Gemini API response did not include an enhanced image")


class TieredEnhancer:
    """Orchestrates tiered upsampling using offline ML first and cloud AI only when needed."""

    def __init__(self, config: EnhanceConfig) -> None:
        self._config = config
        self._opencv = OpenCVEnhancer()
        self._onnx = OnnxSuperResEnhancer(
            model_path=config.onnx_model_path,
            auto_download=config.auto_download_onnx,
            tile_size=config.onnx_tile_size,
            tile_overlap=config.onnx_tile_overlap,
        )
        self._gemini = GeminiEnhancer(
            api_key=config.resolved_gemini_api_key(),
            model=config.gemini_model,
        )

    def enhance_if_needed(
        self,
        image: Image.Image,
        target_ratio: float | None = None,
    ) -> EnhanceResult:
        w, h = image.size
        target_w, target_h, scale = compute_upscale_target(
            w, h, self._config, target_ratio=target_ratio
        )
        if scale <= 1.02:
            return EnhanceResult(
                image=image,
                was_enhanced=False,
                backend="none",
                scale_factor=1.0,
                note="resolution sufficient",
            )

        mode = self._config.mode
        fallback_notes: list[str] = []

        # Decide ordered backend preference chain
        chain: list[ImageEnhancer]
        if mode == "opencv":
            chain = [self._opencv]
        elif mode == "onnx":
            chain = [self._onnx, self._opencv]
        elif mode == "gemini":
            chain = [self._gemini, self._onnx, self._opencv]
        elif mode == "offline":
            if scale < self._config.mild_upscale_threshold:
                chain = [self._opencv]
            else:
                chain = [self._onnx, self._opencv]
        else:  # "auto"
            if scale < self._config.mild_upscale_threshold:
                chain = [self._opencv]
            elif (
                self._config.use_gemini_for_extreme_in_auto
                and scale >= self._config.gemini_extreme_threshold
                and self._config.resolved_gemini_api_key()
            ):
                chain = [self._gemini, self._onnx, self._opencv]
            else:
                chain = [self._onnx, self._opencv]

        for backend in chain:
            try:
                enhanced = backend.enhance(image, target_w, target_h)
                detail = f"upsampled {scale:.2f}x via {backend.name} -> {target_w}x{target_h}"
                if fallback_notes:
                    detail += f" ({'; '.join(fallback_notes)})"
                return EnhanceResult(
                    image=enhanced,
                    was_enhanced=True,
                    backend=backend.name,
                    scale_factor=scale,
                    note=detail,
                )
            except ModelUnavailableError as exc:
                fallback_notes.append(f"{backend.name} skipped: {exc}")

        # OpenCV is always last in chain and cannot raise ModelUnavailableError,
        # but guard defensively anyway.
        return EnhanceResult(
            image=image,
            was_enhanced=False,
            backend="none",
            scale_factor=1.0,
            note="; ".join(fallback_notes) or "enhancement skipped",
        )
