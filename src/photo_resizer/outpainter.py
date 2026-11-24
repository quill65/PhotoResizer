"""Mismatched orientation handling: blurred backdrop fill and Gemini AI outpainting.

Handles portrait <-> landscape aspect ratio expansion without clipping the subject:
- Offline blurred backdrop fill (OpenCV Gaussian blur + subtle vignette + sharp centered original)
- Generative AI outpainting via Gemini API, with feathered compositing of the untouched
  original full-resolution photograph so faces and fine details are never altered.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from photo_resizer.config import DEFAULT_GEMINI_IMAGE_MODEL, EnhanceConfig
from photo_resizer.enhancer import ModelUnavailableError, post_gemini_image_request


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
    subject_center: tuple[float, float] | None = None,
) -> tuple[int, int, int, int]:
    """Compute (canvas_w, canvas_h, offset_x, offset_y) to fit src fully inside target_ratio.

    If subject_center (normalized cx, cy) is provided, shifts the placement within the
    expanded canvas so the subject lands as close to center as possible.
    """
    src_ratio = src_w / src_h
    if src_ratio < target_ratio:
        # Extend left and/or right (e.g., 4:3 -> 16:9 or portrait -> landscape)
        canvas_h = src_h
        canvas_w = max(src_w, int(round(src_h * target_ratio)))
    else:
        # Extend top and/or bottom (e.g., 16:9 -> 4:3 or landscape -> portrait)
        canvas_w = src_w
        canvas_h = max(src_h, int(round(src_w / target_ratio)))

    max_ox = max(0, canvas_w - src_w)
    max_oy = max(0, canvas_h - src_h)

    if subject_center is not None:
        scx, scy = subject_center
        ideal_ox = int(round(canvas_w * 0.5 - scx * src_w))
        ideal_oy = int(round(canvas_h * 0.5 - scy * src_h))
        offset_x = max(0, min(max_ox, ideal_ox))
        offset_y = max(0, min(max_oy, ideal_oy))
    else:
        offset_x = max_ox // 2
        offset_y = max_oy // 2

    return canvas_w, canvas_h, offset_x, offset_y


def create_blurred_backdrop(
    image: Image.Image,
    target_ratio: float,
    subject_center: tuple[float, float] | None = None,
) -> Image.Image:
    """Expand canvas to target_ratio using a softly blurred, color-matched backdrop (offline)."""
    rgb_img = image.convert("RGB")
    src_w, src_h = rgb_img.size
    canvas_w, canvas_h, offset_x, offset_y = compute_expanded_canvas_size(
        src_w, src_h, target_ratio, subject_center=subject_center
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


_SUPPORTED_GEMINI_ASPECT_RATIOS: tuple[tuple[str, float], ...] = (
    ("21:9", 21 / 9),
    ("16:9", 16 / 9),
    ("3:2", 3 / 2),
    ("4:3", 4 / 3),
    ("5:4", 5 / 4),
    ("1:1", 1.0),
    ("4:5", 4 / 5),
    ("3:4", 3 / 4),
    ("2:3", 2 / 3),
    ("9:16", 9 / 16),
)


def nearest_gemini_aspect_ratio(target_ratio: float) -> str:
    """Map any numeric aspect ratio to the closest Gemini imageConfig.aspectRatio string."""
    best_label = "16:9"
    best_diff = float("inf")
    for label, val in _SUPPORTED_GEMINI_ASPECT_RATIOS:
        diff = abs(val - target_ratio) / max(target_ratio, 1e-6)
        if diff < best_diff:
            best_diff = diff
            best_label = label
    return best_label


def _center_fit_to_ratio(image: Image.Image, target_ratio: float) -> Image.Image:
    """Center-crop gently (no stretching!) only if returned aspect ratio differs by >1%."""
    w, h = image.size
    cur_ratio = w / max(1, h)
    if abs(cur_ratio - target_ratio) / max(target_ratio, 1e-6) <= 0.01:
        return image
    if cur_ratio > target_ratio:
        new_w = max(1, int(round(h * target_ratio)))
        left = (w - new_w) // 2
        return image.crop((left, 0, left + new_w, h))
    new_h = max(1, int(round(w / target_ratio)))
    top = (h - new_h) // 2
    return image.crop((0, top, w, top + new_h))


def _estimate_original_alignment(
    original_rgb: np.ndarray,
    ai_rgb: np.ndarray,
) -> tuple[float, float, float] | None:
    """Locate (scale, tx, ty) mapping original_rgb into ai_rgb using SIFT + RANSAC.

    Returns None if geometric alignment is weak, rotated, or doesn't match closely at the seam,
    preventing misaligned vertical cliff discontinuities.
    """
    orig_h, orig_w = original_rgb.shape[:2]
    ai_h, ai_w = ai_rgb.shape[:2]

    # Run SIFT on fast <=800px proxies
    max_dim = 800
    s_orig = min(1.0, max_dim / max(orig_w, orig_h))
    s_ai = min(1.0, max_dim / max(ai_w, ai_h))

    g_orig = cv2.cvtColor(
        cv2.resize(
            original_rgb,
            (max(32, int(round(orig_w * s_orig))), max(32, int(round(orig_h * s_orig)))),
            interpolation=cv2.INTER_AREA,
        ),
        cv2.COLOR_RGB2GRAY,
    )
    g_ai = cv2.cvtColor(
        cv2.resize(
            ai_rgb,
            (max(32, int(round(ai_w * s_ai))), max(32, int(round(ai_h * s_ai)))),
            interpolation=cv2.INTER_AREA,
        ),
        cv2.COLOR_RGB2GRAY,
    )

    try:
        sift = cv2.SIFT_create(nfeatures=1200)
        kp1, des1 = sift.detectAndCompute(g_orig, None)
        kp2, des2 = sift.detectAndCompute(g_ai, None)
    except Exception:
        return None

    if des1 is None or des2 is None or len(kp1) < 16 or len(kp2) < 16:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good = [
        m
        for pair in raw_matches
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.72 * n.distance
    ]
    if len(good) < 20:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    matrix, inliers = cv2.estimateAffinePartial2D(
        pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if matrix is None or inliers is None:
        return None

    inlier_count = int(inliers.sum())
    if inlier_count < 20 or (inlier_count / len(good)) < 0.45:
        return None

    a, tx_p = float(matrix[0, 0]), float(matrix[0, 2])
    c, ty_p = float(matrix[1, 0]), float(matrix[1, 2])
    scale_p = float(np.hypot(a, c))
    angle_deg = abs(float(np.degrees(np.arctan2(c, a))))
    if angle_deg > 1.5 or scale_p <= 0.05:
        return None

    # Convert proxy transform back to full original -> ai_rgb pixel coordinates
    scale = scale_p * (s_orig / s_ai)
    tx = tx_p / s_ai
    ty = ty_p / s_ai

    placed_w = orig_w * scale
    placed_h = orig_h * scale
    # Placed original must fit inside ai_rgb with small tolerance
    if (
        tx < -0.02 * ai_w
        or ty < -0.02 * ai_h
        or tx + placed_w > ai_w * 1.02
        or ty + placed_h > ai_h * 1.02
    ):
        return None

    # Verify photometric consistency along the seam boundary before trusting composite
    ix1 = max(0, int(round(tx)))
    iy1 = max(0, int(round(ty)))
    ix2 = min(ai_w, int(round(tx + placed_w)))
    iy2 = min(ai_h, int(round(ty + placed_h)))
    rw, rh = ix2 - ix1, iy2 - iy1
    if rw < 32 or rh < 32:
        return None

    orig_scaled = cv2.resize(
        original_rgb, (rw, rh), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    ai_roi = ai_rgb[iy1:iy2, ix1:ix2].astype(np.float32)

    # Check mean absolute error on the border band (most sensitive to misalignment seams!)
    band = max(4, min(20, int(round(min(rw, rh) * 0.04))))
    border_diffs: list[float] = []
    if ix1 > 4:
        border_diffs.append(float(np.abs(orig_scaled[:, :band] - ai_roi[:, :band]).mean()))
    if ix2 < ai_w - 4:
        border_diffs.append(float(np.abs(orig_scaled[:, -band:] - ai_roi[:, -band:]).mean()))
    if iy1 > 4:
        border_diffs.append(float(np.abs(orig_scaled[:band, :] - ai_roi[:band, :]).mean()))
    if iy2 < ai_h - 4:
        border_diffs.append(float(np.abs(orig_scaled[-band:, :] - ai_roi[-band:, :]).mean()))

    if border_diffs and max(border_diffs) > 22.0:
        return None

    return (scale, tx, ty)


def align_and_composite_original(
    ai_image: Image.Image,
    original: Image.Image,
    target_w: int,
    target_h: int,
) -> tuple[Image.Image, bool]:
    """Safely blend untouched original pixels into ai_image ONLY if sub-pixel aligned.

    Returns (final_image, did_composite_original). If Gemini reframed or shifted the scene
    so that the boundary wouldn't match seamlessly, returns Gemini's seamless outpaint as-is
    instead of creating a hard vertical/horizontal discontinuity.
    """
    ai_rgb = np.asarray(ai_image.convert("RGB"), dtype=np.uint8)
    orig_rgb = np.asarray(original.convert("RGB"), dtype=np.uint8)

    alignment = _estimate_original_alignment(orig_rgb, ai_rgb)
    if alignment is None:
        out = ai_image.convert("RGB")
        if out.size != (target_w, target_h):
            out = out.resize((target_w, target_h), Image.Resampling.LANCZOS)
        return (out.convert("RGBA") if original.mode == "RGBA" else out), False

    scale, tx, ty = alignment
    ai_h, ai_w = ai_rgb.shape[:2]
    orig_h, orig_w = orig_rgb.shape[:2]

    # Scale AI canvas to target resolution while mapping the aligned original region cleanly
    sx = target_w / ai_w
    sy = target_h / ai_h
    bg = cv2.resize(ai_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4).astype(
        np.float32
    )

    ox1 = max(0, int(round(tx * sx)))
    oy1 = max(0, int(round(ty * sy)))
    ox2 = min(target_w, int(round((tx + orig_w * scale) * sx)))
    oy2 = min(target_h, int(round((ty + orig_h * scale) * sy)))
    rw, rh = ox2 - ox1, oy2 - oy1
    if rw < 32 or rh < 32:
        out = Image.fromarray(np.clip(np.round(bg), 0, 255).astype(np.uint8), mode="RGB")
        return (out.convert("RGBA") if original.mode == "RGBA" else out), False

    orig_fit = cv2.resize(orig_rgb, (rw, rh), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)

    # Smooth raised-cosine feather mask only along interior seams that touch extended AI margins
    mask = np.ones((rh, rw), dtype=np.float32)
    feather_x = max(8, min(64, int(round(rw * 0.05))))
    feather_y = max(8, min(64, int(round(rh * 0.05))))
    ramp_x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, feather_x, dtype=np.float32)))
    ramp_y = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, feather_y, dtype=np.float32)))

    if ox1 > 2:
        mask[:, :feather_x] = np.minimum(mask[:, :feather_x], ramp_x[None, :])
    if ox2 < target_w - 2:
        mask[:, -feather_x:] = np.minimum(mask[:, -feather_x:], ramp_x[::-1][None, :])
    if oy1 > 2:
        mask[:feather_y, :] = np.minimum(mask[:feather_y, :], ramp_y[:, None])
    if oy2 < target_h - 2:
        mask[-feather_y:, :] = np.minimum(mask[-feather_y:, :], ramp_y[::-1][:, None])

    alpha = mask[:, :, None]
    roi = bg[oy1:oy2, ox1:ox2]
    bg[oy1:oy2, ox1:ox2] = orig_fit * alpha + roi * (1.0 - alpha)

    out = Image.fromarray(np.clip(np.round(bg), 0, 255).astype(np.uint8), mode="RGB")
    return (out.convert("RGBA") if original.mode == "RGBA" else out), True


class GeminiOutpainter:
    """Generative canvas expander using Gemini native aspectRatio image outpainting."""

    def __init__(self, api_key: str | None, model: str = DEFAULT_GEMINI_IMAGE_MODEL) -> None:
        self._api_key = api_key
        self._model = model
        self.last_composited_original: bool = False

    def outpaint(
        self,
        image: Image.Image,
        target_ratio: float,
        subject_center: tuple[float, float] | None = None,
    ) -> Image.Image:
        del subject_center  # Gemini native outpaint centers and extends seamlessly without bars
        if not self._api_key:
            raise ModelUnavailableError("GEMINI_API_KEY is not configured")

        src_w, src_h = image.size
        canvas_w, canvas_h, _, _ = compute_expanded_canvas_size(src_w, src_h, target_ratio)
        horizontal_extend = canvas_w > src_w
        ratio_str = nearest_gemini_aspect_ratio(target_ratio)

        # Send the clean, UN-PADDED original photo (no harsh white bars!) so Gemini
        # natively extends leaves, petals, sky, and bokeh seamlessly across the edges.
        upload_max = 1536
        u_scale = min(1.0, upload_max / max(src_w, src_h))
        if u_scale < 1.0:
            upload_img = image.convert("RGB").resize(
                (max(64, int(round(src_w * u_scale))), max(64, int(round(src_h * u_scale)))),
                Image.Resampling.LANCZOS,
            )
        else:
            upload_img = image.convert("RGB")

        buf = io.BytesIO()
        upload_img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        direction = "left and right sides" if horizontal_extend else "top and bottom edges"
        prompt = (
            f"Zoom out and seamlessly outpaint this photograph to fill a {ratio_str} frame by "
            f"continuing the natural scenery, leaves, stems, branches, lighting, shadows, and "
            f"depth-of-field blur smoothly beyond the {direction}. Keep the original subject "
            "(animal, insect, flower, or person) centered, completely intact, and faithful to "
            "the original photograph. Produce one continuous, natural photograph with zero "
            "borders, zero frames, and zero visible seams."
        )

        generated_bytes = self._call_gemini_rest(
            png_bytes, prompt, aspect_ratio=ratio_str, image_size="2K"
        )
        with Image.open(io.BytesIO(generated_bytes)) as generated:
            ai_img = _center_fit_to_ratio(generated.convert("RGB"), target_ratio)

        # Preserve full-resolution original pixels ONLY if SIFT+RANSAC confirms sub-pixel
        # alignment with zero seam mismatch; otherwise keep Gemini's seamless output cleanly.
        final_img, did_composite = align_and_composite_original(
            ai_image=ai_img,
            original=image,
            target_w=canvas_w,
            target_h=canvas_h,
        )
        self.last_composited_original = did_composite
        return final_img

    def _call_gemini_rest(
        self,
        png_bytes: bytes,
        prompt: str,
        *,
        aspect_ratio: str | None = None,
        image_size: str | None = None,
    ) -> bytes:
        return post_gemini_image_request(
            api_key=self._api_key or "",
            model=self._model,
            png_bytes=png_bytes,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
        )


def expand_canvas_blur_fill(
    image: Image.Image,
    target_ratio: float,
    *,
    subject_center: tuple[float, float] | None = None,
    reason_prefix: str = "mismatched orientation",
) -> OutpaintResult:
    """Expand photo canvas using offline blurred backdrop fill."""
    src_w, src_h = image.size
    expanded = create_blurred_backdrop(image, target_ratio, subject_center=subject_center)
    out_w, out_h = expanded.size
    return OutpaintResult(
        image=expanded,
        method="blur_fill",
        tool_detail="OpenCV/Pillow (blurred_backdrop_fill)",
        note=f"{reason_prefix}: expanded {src_w}x{src_h} -> {out_w}x{out_h} via blurred fill",
    )


def expand_canvas_ai_outpaint(
    image: Image.Image,
    target_ratio: float,
    enhance_config: EnhanceConfig,
    *,
    subject_center: tuple[float, float] | None = None,
    reason_prefix: str = "mismatched orientation",
) -> OutpaintResult:
    """Expand photo canvas via Gemini AI outpainting with offline blurred fill fallback."""
    src_w, src_h = image.size
    outpainter = GeminiOutpainter(
        api_key=enhance_config.resolved_gemini_api_key(),
        model=enhance_config.gemini_model,
    )
    try:
        expanded = outpainter.outpaint(image, target_ratio, subject_center=subject_center)
        out_w, out_h = expanded.size
        suffix = (
            " + Aligned Original Composite"
            if outpainter.last_composited_original
            else " (Seamless Native Outpaint)"
        )
        return OutpaintResult(
            image=expanded,
            method="ai_outpaint",
            tool_detail=f"Gemini AI Outpaint ({enhance_config.gemini_model}){suffix}",
            note=(
                f"{reason_prefix}: outpainted {src_w}x{src_h} -> {out_w}x{out_h} "
                f"via Gemini ({enhance_config.gemini_model})"
            ),
        )
    except ModelUnavailableError as exc:
        fallback = expand_canvas_blur_fill(
            image,
            target_ratio,
            subject_center=subject_center,
            reason_prefix=reason_prefix,
        )
        return OutpaintResult(
            image=fallback.image,
            method="blur_fill",
            tool_detail=fallback.tool_detail,
            note=f"{fallback.note} (ai_outpaint skipped: {exc})",
        )
