"""Fill-value and scale/offset decoding for raw arrays (h5py / pyhdf paths)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _scalar(v):
    arr = np.asarray(v).ravel()
    if arr.dtype.kind in "SUO":
        try:
            return float(arr[0].decode() if isinstance(arr[0], bytes) else arr[0])
        except (ValueError, IndexError):
            return None
    return arr[0].item() if arr.size else None


def decode(raw: np.ndarray, attrs: Mapping, *, hdf4_convention: bool = False) -> np.ndarray:
    """Apply _FillValue / missing_value / valid_range masking, then scale and offset.

    CF:   value = raw * scale_factor + add_offset
    HDF4 (MODIS): value = scale_factor * (raw - add_offset)
    """
    if raw.dtype.kind not in "iuf":
        return raw
    out = raw.astype("float64")
    lower = {str(k).lower(): v for k, v in attrs.items()}
    for key in ("_fillvalue", "missing_value", "fillvalue", "_fill_value"):
        if key in lower and (fv := _scalar(lower[key])) is not None:
            out[raw == fv] = np.nan
    if "valid_range" in lower:
        vr = np.asarray(lower["valid_range"]).ravel()
        if vr.size == 2 and vr.dtype.kind in "iuf":
            out[(raw < vr[0]) | (raw > vr[1])] = np.nan
    for key, cmp in (("valid_min", np.less), ("valid_max", np.greater)):
        if key in lower and (lim := _scalar(lower[key])) is not None:
            out[cmp(raw, lim)] = np.nan
    scale = _scalar(lower.get("scale_factor", 1.0)) or 1.0
    offset = _scalar(lower.get("add_offset", 0.0)) or 0.0
    if hdf4_convention:
        return scale * (out - offset)
    return out * scale + offset


def clean_attrs(attrs: Mapping, limit: int = 40) -> dict:
    """JSON-friendly attribute dict (bytes decoded, arrays shortened)."""
    out = {}
    for i, (k, v) in enumerate(attrs.items()):
        if i >= limit:
            out["..."] = f"{len(attrs) - limit} more"
            break
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
        elif isinstance(v, np.ndarray):
            v = v.tolist() if v.size <= 8 else f"array{v.shape}"
            if isinstance(v, list):
                v = [x.decode("utf-8", "replace") if isinstance(x, bytes) else x for x in v]
        elif isinstance(v, np.generic):
            v = v.item()
        s = str(v)
        out[str(k)] = v if len(s) <= 300 and isinstance(v, int | float | str | list) else s[:300]
    return out
