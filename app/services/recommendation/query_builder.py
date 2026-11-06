"""Template truy vấn dùng chung cho runtime và encode offline.
Thay đổi chuỗi template sẽ làm query signature thay đổi và cần encode lại.
"""

import hashlib

from app.data.body_shapes import canonicalize_recommendation_shape

SEASON_KEYWORDS: dict[str, str] = {
    "Light Spring": "warm light peachy coral pastel clean menswear soft contrast fresh",
    "True Spring": "warm golden vibrant green orange yellow energetic menswear crisp cheerful",
    "Bright Spring": "warm vivid bright saturated bold colorful menswear high energy contrast",
    "Light Summer": "cool soft pastel lavender baby blue rose powder airy menswear refined",
    "True Summer": "cool muted dusty rose berry mauve elegant menswear soft tonal tailoring",
    "Soft Summer": "cool muted neutral taupe sage soft blended menswear low contrast relaxed tailoring",
    "Soft Autumn": "warm muted olive khaki camel beige earthy soft menswear textured understated",
    "True Autumn": "warm rich earthy terracotta rust brown burnt orange menswear rugged classic",
    "Deep Autumn": "warm deep dark brown forest green burgundy menswear rich contrast mature",
    "Deep Winter": "cool deep dark navy black jewel tone dramatic menswear sharp contrast sleek",
    "True Winter": "cool bold contrast black white red icy pure menswear crisp monochrome clean",
    "Bright Winter": "cool vivid bright cobalt electric saturated menswear glossy high contrast statement",
}

BODY_KEYWORDS: dict[str, str] = {
    "Trapezoid": "menswear tailored fit balanced torso clean shoulder line structured blazer tapered chinos regular trousers",
    "Rectangle": "layered menswear overshirt bomber trucker jacket straight fit trousers texture and structure",
    "Inverted Triangle": "menswear V-neck raglan top straight leg wide leg trousers balanced lower body clean drape",
    "Triangle": "menswear structured shoulder jacket crew neck darker top lighter bottom straight trousers balanced proportion",
    "Oval": "menswear vertical line outfit open layer mid-rise straight leg clean minimal streamlined torso",
}

OCCASION_KEYWORDS: dict[str, str] = {
    "daily": "mens casual everyday comfortable relaxed streetwear clean sneakers overshirt",
    "work": "mens office professional business casual blazer chinos loafers polished neat",
    "party": "mens evening smart casual sharp clean dark tonal outfit refined texture",
    "date": "mens date night smart casual polished minimal style soft tailoring elevated basics",
}

SEASON_COLOR_FAMILY: dict[str, set[str]] = {
    "Spring": {"warm_bright", "earth", "rich"},
    "Summer": {"soft_cool", "cool", "neutral"},
    "Autumn": {"earth", "rich", "neutral"},
    "Winter": {"high_contrast", "cool", "rich"},
}

BODY_SHAPE_COMPATIBILITY: dict[str, set[str]] = {
    "Trapezoid": {"trapezoid", "rectangle"},
    "Rectangle": {"rectangle", "trapezoid"},
    "Inverted Triangle": {"inverted triangle", "trapezoid"},
    "Triangle": {"triangle", "oval"},
    "Oval": {"oval", "triangle"},
}


def _all_query_templates() -> list[str]:
    queries: list[str] = []
    for season in SEASON_KEYWORDS:
        for body_shape in BODY_KEYWORDS:
            for occasion in OCCASION_KEYWORDS:
                queries.append(build_outfit_query(season, body_shape, occasion))
    return queries


def _current_query_signature() -> str:
    payload = "\n".join(sorted(_all_query_templates())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _primary_season(sub_season: str) -> str:
    normalized = str(sub_season or "").strip().lower()
    if "spring" in normalized:
        return "Spring"
    if "summer" in normalized:
        return "Summer"
    if "autumn" in normalized:
        return "Autumn"
    if "winter" in normalized:
        return "Winter"
    return ""


def build_outfit_query(
    color_sub_season: str,
    body_shape: str,
    occasion: str = "daily",
) -> str:
    """Build a menswear-focused semantic search query."""
    season_kw = SEASON_KEYWORDS.get(color_sub_season, "refined menswear color palette")
    canonical_shape = canonicalize_recommendation_shape(body_shape)
    body_kw = BODY_KEYWORDS.get(
        canonical_shape, "menswear flattering well-fitted proportions"
    )
    occasion_kw = OCCASION_KEYWORDS.get(occasion, "casual everyday menswear")
    return (
        f"A men's outfit with {season_kw}, "
        f"suited for a {canonical_shape} body using {body_kw}, "
        f"{occasion_kw}, ecommerce product photo, realistic styling, wearable proportions"
    )


def _parse_bias_tags(value: str | None) -> set[str]:
    if not value:
        return set()
    raw_parts = [
        part.strip()
        for part in str(value).replace(",", "|").replace("/", "|").split("|")
        if part.strip()
    ]
    return {part.casefold() for part in raw_parts}
