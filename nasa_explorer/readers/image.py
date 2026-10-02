"""Plain images (PNG/JPEG/TIFF/GIF/BMP/WebP) via Pillow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from ..core import ReadOptions, ReadResult
from ..registry import reader

MAX_PIXELS = 40_000_000


def _gps(ifd) -> dict | None:
    """EXIF GPS IFD -> decimal degrees (tags 1-4: lat ref, lat, lon ref, lon)."""
    try:
        lat = sum(float(v) / 60**i for i, v in enumerate(ifd[2]))
        lon = sum(float(v) / 60**i for i, v in enumerate(ifd[4]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    lat *= -1 if ifd.get(1) == "S" else 1
    lon *= -1 if ifd.get(3) == "W" else 1
    return {"lat": round(lat, 5), "lon": round(lon, 5)}


@reader(
    "image",
    category="Images",
    extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp", ".webp"),
    magic=(b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"BM", b"II*\x00", b"MM\x00*", b"RIFF"),
    requires=("PIL",),
)
def read_image(path: Path, opts: ReadOptions) -> ReadResult:
    from PIL import ExifTags, Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        meta = {
            "mode": im.mode,
            "width": im.width,
            "height": im.height,
            "format": im.format,
            "frames": getattr(im, "n_frames", 1),
        }
        raw_exif = im.getexif()
        exif = {ExifTags.TAGS.get(k, str(k)): str(v)[:100] for k, v in raw_exif.items()}
        if exif:
            meta["exif"] = dict(list(exif.items())[:30])
        if taken := raw_exif.get_ifd(0x8769).get(36867) or raw_exif.get(306):
            meta["taken"] = str(taken)
        if gps := _gps(raw_exif.get_ifd(0x8825)):
            meta["gps"] = gps
        if im.width * im.height > MAX_PIXELS:
            im.thumbnail((4096, 4096))
            meta["downsampled_to"] = [im.width, im.height]
        if im.mode in ("P", "1", "CMYK", "LA", "PA"):
            im = im.convert("RGBA" if "A" in im.mode else "RGB")
        arr = np.asarray(im)
    if arr.ndim == 2:
        arr = arr[..., None]
    bands = (
        list(meta["mode"])
        if len(meta["mode"]) == arr.shape[-1] and meta["mode"].isalpha() and meta["mode"].isupper()
        else [f"b{i}" for i in range(arr.shape[-1])]
    )
    da = xr.DataArray(arr, dims=("y", "x", "band"), coords={"band": bands})
    return ReadResult("image", xr.Dataset({"image": da}), meta)
