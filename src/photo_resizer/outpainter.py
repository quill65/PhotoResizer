"""Mismatched orientation handling: blurred backdrop fill and Gemini AI outpainting.

Handles portrait <-> landscape aspect ratio expansion without clipping the subject:
- Offline blurred backdrop fill (OpenCV Gaussian blur + subtle vignette + sharp centered original)
- Generative AI outpainting via Gemini API, with feathered compositing of the untouched
  original full-resolution photograph so faces and fine details are never altered.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import cv2
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from photo_resizer.config import EnhanceConfig
from photo_resizer.enhancer import ModelUnavailableError


@dataclass(frozen=True)
class OutpaintResult:
    """Result from canvas expansion (blurred fill or AI outpainting)."""

    image: Image.Image
    method: str  # "blur_fill" | "ai_outpaint"
    tool_detail: str
    note: str


def is_orientation_mismatch(src_w: int, src_h: int, target_ratio: float) -> bool:
    """Return True when source and target have opposite orientations (portrait vs landscape)."""
    if src_w <= 0 or src_h <= 0 or target_ratio <= 0:
        return False
    src_ratio = src_w / src_h
    src_is_portrait = src_ratio < 0.98
    src_is_landscape = src_ratio > 1.02
    tgt_is_portrait = target_ratio < 0.98
    tgt_is_landscape = target_ratio > 1.02
    return (src_is_portrait and tgt_is_landscape) or (src_is_landscape and tgt_is_portrait)


def compute_expanded_canvas_size(
    src_w: int,
    src_h: int,
    target_ratio: float,
) -> tuple[int, int, int, int]:
    """Compute (canvas_w, canvas_h, offset_x, offset_y) to fit src fully inside target_ratio."""
    src_ratio = src_w / src_h
    if src_ratio < target_ratio:
        # Extend left and right (e.g., portrait -> landscape)
        canvas_h = src_h
        canvas_w = max(src_w, int(round(src_h * target_ratio)))
    else:
        # Extend top and bottom (e.g., landscape -> portrait)
        canvas_w = src_w
        canvas_h = max(src_h, int(round(src_w / target_ratio)))

    offset_x = (canvas_w - src_w) // 2
    offset_y = (canvas_h - src_h) // 2
    return canvas_w, canvas_h, offset_x, offset_y


def create_blurred_backdrop(image: Image.Image, target_ratio: float) -> Image.Image:
    """Expand canvas to target_ratio using a softly blurred, color-matched backdrop (offline)."""
    rgb_img = image.convert("RGB")
    src_w, src_h = rgb_img.size
    canvas_w, canvas_h, offset_x, offset_y = compute_expanded_canvas_size(
        src_w, src_h, target_ratio
    )

    # Scale source in 'cover' mode so backdrop completely fills the target canvas
    scale_cover = max(canvas_w / src_w, canvas_h / src_h)
    cover_w = max(canvas_w, int(round(src_w * scale_cover)))
    cover_h = max(canvas_h, int(round(src_h * scale_cover)))
    cover = rgb_img.resize((cover_w, cover_h), Image.Resampling.BILINEAR)
    left = (cover_w - canvas_w) // 2
    top = (cover_h - canvas_h) // 2
    bg = cover.crop((left, top, left + canvas_w, top + canvas_h))

    # Fast, smooth heavy blur on a downscaled proxy, then upscale smoothly
    proxy_max = 480
    p_scale = min(1.0, proxy_max / max(canvas_w, canvas_h))
    pw = max(16, int(round(canvas_w * p_scale)))
    ph = max(16, int(round(canvas_h * p_scale)))
    bg_small = np.asarray(bg.resize((pw, ph), Image.Resampling.BILINEAR), dtype=np.uint8)

    ksize = max(15, (int(round(max(pw, ph) * 0.08)) | 1))
    blurred = cv2.GaussianBlur(bg_small, (ksize, ksize), 0)
    # Slightly darken backdrop so foreground subject stands out cleanly
    darkened = np.clip(blurred.astype(np.float32) * 0.72, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(darkened, mode="RGB").resize(
        (canvas_w, canvas_h), Image.Resampling.BICUBIC
    )

    # Subtle drop shadow behind the sharp original photo
    shadow_pad = max(6, int(round(min(canvas_w, canvas_h) * 0.015)))
    shadow_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow_layer)
    draw.rectangle(
        [
            max(0, offset_x - shadow_pad // 2),
            max(0, offset_y - shadow_pad // 2),
            min(canvas_w, offset_x + src_w + shadow_pad // 2),
            min(canvas_h, offset_y + src_h + shadow_pad // 2),
        ],
        fill=(0, 0, 0, 110),
    )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=shadow_pad))
    canvas_rgba = canvas.convert("RGBA")
    canvas_rgba = Image.alpha_composite(canvas_rgba, shadow_layer)
    canvas_rgba.paste(image.convert("RGBA"), (offset_x, offset_y))

    return canvas_rgba if image.mode == "RGBA" else canvas_rgba.convert("RGB")


def composite_original_with_feather(
    ai_background: Image.Image,
    original: Image.Image,
    offset_x: int,
    offset_y: int,
    horizontal_extend: bool,
) -> Image.Image:
    """Blend the untouched original photo onto the AI-extended background using a soft seam."""
    canvas_w, canvas_h = ai_background.size
    src_w, src_h = original.size

    bg_arr = np.asarray(ai_background.convert("RGB"), dtype=np.float32)
    orig_arr = np.asarray(original.convert("RGB"), dtype=np.float32)

    # Build a feathered alpha mask over the original region so only the seam transitions softly
    # while >90% of the original photo remains 100% untouched.
    mask = np.ones((src_h, src_w), dtype=np.float32)
    if horizontal_extend:
        feather = max(4, min(48, int(round(src_w * 0.035))))
        ramp = np.linspace(0.0, 1.0, feather, dtype=np.float32)
        if offset_x > 0:
            mask[:, :feather] = np.minimum(mask[:, :feather], ramp[None, :])
        if offset_x + src_w < canvas_w:
            mask[:, -feather:] = np.minimum(mask[:, -feather:], ramp[::-1][None, :])
    else:
        feather = max(4, min(48, int(round(src_h * 0.035))))
        ramp = np.linspace(0.0, 1.0, feather, dtype=np.float32)
        if offset_y > 0:
            mask[:feather, :] = np.minimum(mask[:feather, :], ramp[:, None])
        if offset_y + src_h < canvas_h:
            mask[-feather:, :] = np.minimum(mask[-feather:, :], ramp[::-1][:, None])

    alpha = mask[:, :, None]
    roi = bg_arr[offset_y : offset_y + src_h, offset_x : offset_x + src_w]
    blended = orig_arr * alpha + roi * (1.0 - alpha)
    bg_arr[offset_y : offset_y + src_h, offset_x : offset_x + src_w] = blended

    out = Image.fromarray(np.clip(np.round(bg_arr), 0, 255).astype(np.uint8), mode="RGB")
    if original.mode == "RGBA":
        return out.convert("RGBA")
    return out


class GeminiOutpainter:
    """Generative canvas expander using Gemini image editing API."""

    def __init__(self, api_key: str | None, model: str = "gemini-2.5-flash-image") -> None:
        self._api_key = api_key
        self._model = model

    def outpaint(
        self,
        image: Image.Image,
        target_ratio: float,
    ) -> Image.Image:
        if not self._api_key:
            raise ModelUnavailableError("GEMINI_API_KEY is not configured")

        src_w, src_h = image.size
        canvas_w, canvas_h, offset_x, offset_y = compute_expanded_canvas_size(
            src_w, src_h, target_ratio
        )
        horizontal_extend = canvas_w > src_w

        # Build a clean padded guide canvas at <=1536px long edge for the Gemini request
        guide_max = 1536
        g_scale = min(1.0, guide_max / max(canvas_w, canvas_h))
        gw = max(64, int(round(canvas_w * g_scale)))
        gh = max(64, int(round(canvas_h * g_scale)))
        sw = max(32, int(round(src_w * g_scale)))
        sh = max(32, int(round(src_h * g_scale)))
        gx = (gw - sw) // 2
        gy = (gh - sh) // 2

        guide = Image.new("RGB", (gw, gh), color=(255, 255, 255))
        guide.paste(image.convert("RGB").resize((sw, sh), Image.Resampling.LANCZOS), (gx, gy))

        buf = io.BytesIO()
        guide.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        direction = (
            "empty white bars on the left and right sides"
            if horizontal_extend
            else "empty white bars on the top and bottom"
        )
        prompt = (
            f"Seamlessly outpaint and extend the photograph to naturally fill the {direction}, "
            "matching the exact lighting, camera perspective, colors, depth-of-field blur, and "
            "surrounding environment. Keep the original central photograph completely intact "
            "without altering any people, faces, objects, or proportions, and maintain the exact "
            "output canvas aspect ratio with no borders or frames."
        )

        generated_bytes = self._call_gemini_rest(png_bytes, prompt)
        with Image.open(io.BytesIO(generated_bytes)) as generated:
            ai_bg = generated.convert("RGB")

        if ai_bg.size != (canvas_w, canvas_h):
            ai_bg = ai_bg.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)

        # Feather-composite untouched original pixels back onto the AI canvas
        return composite_original_with_feather(
            ai_background=ai_bg,
            original=image,
            offset_x=offset_x,
            offset_y=offset_y,
            horizontal_extend=horizontal_extend,
        )

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
            raise ModelUnavailableError(f"Gemini outpaint API request failed: {exc}") from exc

        for candidate in data.get("candidates", []):
            parts = candidate.get("content", {}).get("parts", [])
            for part in parts:
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    return base64.b64decode(inline["data"])

        raise ModelUnavailableError("Gemini outpaint response did not include an image")


def expand_canvas_blur_fill(image: Image.Image, target_ratio: float) -> OutpaintResult:
    """Expand mismatched-orientation photo using offline blurred backdrop fill."""
    src_w, src_h = image.size
    expanded = create_blurred_backdrop(image, target_ratio)
    out_w, out_h = expanded.size
    return OutpaintResult(
        image=expanded,
        method="blur_fill",
        tool_detail="OpenCV/Pillow (blurred_backdrop_fill)",
        note=f"mismatched orientation expanded {src_w}x{src_h} -> {out_w}x{out_h} via blurred fill",
    )


def expand_canvas_ai_outpaint(
    image: Image.Image,
    target_ratio: float,
    enhance_config: EnhanceConfig,
) -> OutpaintResult:
    """Expand mismatched-orientation photo via Gemini AI outpainting with offline blur fallback."""
    src_w, src_h = image.size
    outpainter = GeminiOutpainter(
        api_key=enhance_config.resolved_gemini_api_key(),
        model=enhance_config.gemini_model,
    )
    try:
        expanded = outpainter.outpaint(image, target_ratio)
        out_w, out_h = expanded.size
        return OutpaintResult(
            image=expanded,
            method="ai_outpaint",
            tool_detail=(
                f"Gemini AI Outpaint ({enhance_config.gemini_model}) + "
                "Feathered Original Composite"
            ),
            note=(
                f"mismatched orientation outpainted {src_w}x{src_h} -> {out_w}x{out_h} "
                f"via Gemini ({enhance_config.gemini_model})"
            ),
        )
    except ModelUnavailableError as exc:
        fallback = expand_canvas_blur_fill(image, target_ratio)
        return OutpaintResult(
            image=fallback.image,
            method="blur_fill",
            tool_detail=fallback.tool_detail,
            note=f"{fallback.note} (ai_outpaint skipped: {exc})",
        )
