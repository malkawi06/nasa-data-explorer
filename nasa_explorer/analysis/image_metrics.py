"""Measurements of a plain image computed by code, so the AI's visual claims have numbers to be
checked against. Shares are percentages of all pixels; colour thresholds are deliberately simple
and their names say what they measure (e.g. "green-dominant"), not what the pixels are."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np

MAX_SIDE = 1024  # measure on a downsampled copy: fast and enough for shares and colours
COLOUR_MARGIN = 0.04  # a channel must beat the others by this much to "dominate" a pixel
MIN_COLOUR_SHARE = 0.001


def _rgb01(arr: np.ndarray) -> np.ndarray:
    """(y, x, band) -> float RGB in [0, 1]; grey is repeated, alpha dropped."""
    a = arr.astype(np.float64)
    if np.issubdtype(arr.dtype, np.integer):
        a /= np.iinfo(arr.dtype).max
    else:
        hi = np.nanmax(a) if np.isfinite(a).any() else 1.0
        a /= hi if hi > 0 else 1.0
    a = np.nan_to_num(np.clip(a, 0, 1))
    if a.shape[-1] in (1, 2):
        a = np.repeat(a[..., :1], 3, axis=-1)
    return a[..., :3]


def _downsample(arr: np.ndarray) -> np.ndarray:
    step = max(1, int(np.ceil(max(arr.shape[:2]) / MAX_SIDE)))
    return arr[::step, ::step]


def measure(arr: np.ndarray) -> dict:
    rgb = _rgb01(_downsample(arr))
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.where(mx > 0, mx, 1), 0)
    rg, yb = r - g, 0.5 * (r + g) - b  # Hasler & Suesstrunk colourfulness
    colourfulness = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())
    lap = (
        -4 * lum[1:-1, 1:-1] + lum[:-2, 1:-1] + lum[2:, 1:-1] + lum[1:-1, :-2] + lum[1:-1, 2:]
    ) * 255
    gy, gx = np.gradient(lum)

    q = (rgb * 3.999).astype(int)  # 4 levels per channel -> 64 colour bins
    codes = q[..., 0] * 16 + q[..., 1] * 4 + q[..., 2]
    shares = np.bincount(codes.ravel(), minlength=64) / codes.size
    order = np.argsort(shares)[::-1]
    dominant = {}
    for c in order[:5]:
        if shares[c] < 0.01:
            break
        sel = codes == c
        mean = (rgb[sel].mean(0) * 255).round().astype(int)
        dominant["#{:02x}{:02x}{:02x}".format(*mean)] = round(100 * float(shares[c]), 1)

    def pct(mask: np.ndarray) -> float:
        return round(100 * float(mask.mean()), 1)

    m = {
        "brightness_pct": round(100 * float(lum.mean()), 1),
        "contrast_pct": round(100 * float(lum.std()), 1),
        "dark_clipped_pct": pct(lum < 0.02),
        "bright_clipped_pct": pct(lum > 0.98),
        "saturation_pct": round(100 * float(sat.mean()), 1),
        "colourfulness": round(100 * float(colourfulness), 1),
        "sharpness": round(float(lap.var()), 1) if lap.size else 0.0,
        "edge_density_pct": pct(np.hypot(gx, gy) > 0.1),
        "distinct_colours": int((shares >= MIN_COLOUR_SHARE).sum()),
        "largest_colour_share_pct": round(100 * float(shares.max()), 1),
        "white_low_saturation_pct": pct((lum > 0.75) & (sat < 0.15)),
        "green_dominant_pct": pct((g > r + COLOUR_MARGIN) & (g > b + COLOUR_MARGIN)),
        "blue_dominant_pct": pct((b > r + COLOUR_MARGIN) & (b > g + COLOUR_MARGIN)),
        "dark_pct": pct(lum < 0.1),
        "dominant_colours": dominant,
    }
    m["likely_type"] = _likely_type(m)
    return m


def _likely_type(m: dict) -> str:
    """A rough label from the measurements; the AI may refine it from the picture itself."""
    if m["saturation_pct"] < 3:
        if m["dark_pct"] > 60:
            return "grayscale, mostly dark (astronomy or night scene?)"
        return "grayscale"
    if m["largest_colour_share_pct"] > 35 and m["distinct_colours"] < 40:
        return "flat colours (chart, diagram or map?)"
    if m["dark_pct"] > 60:
        return "mostly dark (astronomy or night scene?)"
    return "continuous-tone (photo or satellite scene?)"


def jpeg_b64(path: Path, side: int = MAX_SIDE) -> str | None:
    """Downscaled 8-bit JPEG of an image file, base64-encoded (None if it cannot be read)."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            if im.mode in ("P", "1", "CMYK", "LA", "PA"):
                im = im.convert("RGBA" if "A" in im.mode else "RGB")
            im.thumbnail((side, side))
            arr = np.asarray(im)
    except (OSError, ValueError):
        return None
    if arr.ndim == 2:
        arr = arr[..., None]
    rgb = (_rgb01(arr) * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()
