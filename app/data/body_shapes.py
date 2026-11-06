"""Taxonomy nam và alias lịch sử; không suy ra số đo từ tên dáng."""

MALE_BODY_SHAPES = ("Trapezoid", "Rectangle", "Inverted Triangle", "Triangle", "Oval")
COMMON_ALIASES = {name.lower(): name for name in MALE_BODY_SHAPES}
COMMON_ALIASES.update({"hourglass": "Trapezoid", "soft hourglass": "Trapezoid"})
INFERENCE_ALIASES = {
    **COMMON_ALIASES,
    "hinh thang": "Trapezoid",
    "chu nhat": "Rectangle",
    "tam giac nguoc": "Inverted Triangle",
    "tam giac": "Triangle",
}


def canonicalize_inferred_shape(value: object) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if not raw:
        return None
    return INFERENCE_ALIASES.get(raw, str(value).strip())


def canonicalize_recommendation_shape(shape: str) -> str:
    raw = str(shape or "").strip().lower()
    return COMMON_ALIASES.get(raw, str(shape).strip())
