import math


DEFAULT_DENSITY_SCALES = {
    "mvtec3d": (0.03, 0.07, 0.11, 0.15),
    "eyecandies": (0.02, 0.03, 0.04, 0.05),
}


def resolve_density_scales(dataset_type, override=None):
    """Return the four MDFE radius multipliers in their search order."""
    if override is None:
        return DEFAULT_DENSITY_SCALES[dataset_type]
    if isinstance(override, str):
        override = override.split(",")
    message = "expected four finite, positive density scales"
    try:
        scales = tuple(float(value) for value in override)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if len(scales) != 4 or any(not math.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError(message)
    return scales
