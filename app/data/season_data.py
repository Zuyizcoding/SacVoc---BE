"""
12-Season Lookup Table — Palette + Makeup Guide.

Tính năng #20: Palette server-side override — Gemini KHÔNG tự sinh hex.
Sau khi nhận kết quả từ Gemini, server thay palette bằng dữ liệu cứng từ file này.

Cách dùng trong color_service.py:
    from app.data.season_data import SEASON_DATA
    result["palette"] = SEASON_DATA[result["sub_season"]]["palette"]
"""
from typing import TypedDict


class MakeupData(TypedDict):
    lip: list[str]
    eye: list[str]
    hair: list[str]


class MenswearData(TypedDict):
    tops: list[str]
    bottoms: list[str]
    accessories: list[str]


class SeasonEntry(TypedDict):
    primary: str
    palette: list[str]       # 6 hex codes chuẩn
    makeup: MakeupData
    gradient_from: str       # Cho SeasonGradientCard (frontend)
    gradient_to: str         # Cho SeasonGradientCard (frontend)


MENSWEAR_BY_SUBSEASON: dict[str, MenswearData] = {
    "Light Spring": {
        "tops": ["Kem sáng", "xanh mint nhạt", "đào ấm rất nhẹ"],
        "bottoms": ["Be sáng", "stone", "xanh navy nhạt"],
        "accessories": ["Sneaker trắng ngà", "kim loại vàng nhạt", "đồng hồ dây da màu tan"],
    },
    "True Spring": {
        "tops": ["Coral ấm", "vàng ấm vừa phải", "xanh lá tươi sạch"],
        "bottoms": ["Khaki ấm", "navy sạch", "nâu sáng"],
        "accessories": ["Đồng hồ dây da nâu sáng", "kim loại vàng", "loafer màu tan"],
    },
    "Bright Spring": {
        "tops": ["Trắng sạch", "xanh cobalt sáng", "cam đỏ rõ sắc"],
        "bottoms": ["Navy đậm", "be sáng", "than chì nhạt"],
        "accessories": ["Sneaker trắng sắc nét", "kim loại bóng", "kính gọng đen gọn"],
    },
    "Light Summer": {
        "tops": ["Xanh baby", "xám lạnh sáng", "hồng bụi sáng"],
        "bottoms": ["Ghi sáng", "navy mềm", "be lạnh"],
        "accessories": ["Sneaker trắng lạnh", "kim loại bạc", "thắt lưng nâu tro sáng"],
    },
    "True Summer": {
        "tops": ["Xanh dusty", "mauve lạnh", "xám xanh trung tính"],
        "bottoms": ["Charcoal mềm", "navy muted", "stone lạnh"],
        "accessories": ["Bạc xước mờ", "giày suede xám", "đồng hồ mặt bạc lạnh"],
    },
    "Soft Summer": {
        "tops": ["Taupe lạnh", "sage dịu", "xám khói mềm"],
        "bottoms": ["Nâu tro", "ghi trung tính", "navy xám"],
        "accessories": ["Da lộn xám nâu", "kim loại bạc mờ", "túi canvas tông trầm"],
    },
    "Soft Autumn": {
        "tops": ["Olive muted", "camel nhạt", "be ấm trầm"],
        "bottoms": ["Khaki đậm", "nâu chocolate vừa", "denim wash đậm"],
        "accessories": ["Da nâu lì", "kim loại đồng", "boots suede nâu mềm"],
    },
    "True Autumn": {
        "tops": ["Terracotta", "olive đậm", "camel rõ sắc"],
        "bottoms": ["Nâu đất", "khaki đậm", "indigo trầm"],
        "accessories": ["Thắt lưng da nâu đậm", "đồng hồ mặt đồng", "boots da nâu"],
    },
    "Deep Autumn": {
        "tops": ["Forest green", "nâu espresso", "burgundy ấm"],
        "bottoms": ["Than nâu", "olive đậm", "denim đậm không wash mạnh"],
        "accessories": ["Da nâu sẫm", "kim loại đồng tối", "boots nâu chocolate"],
    },
    "Deep Winter": {
        "tops": ["Navy đậm", "đen", "burgundy lạnh"],
        "bottoms": ["Charcoal", "đen", "xám lạnh đậm"],
        "accessories": ["Kim loại bạc", "giày da đen", "đồng hồ mặt tối giản"],
    },
    "True Winter": {
        "tops": ["Trắng sắc nét", "đen", "navy lạnh sạch"],
        "bottoms": ["Than chì", "đen", "xám lạnh"],
        "accessories": ["Bạc bóng", "derby đen", "kính gọng đen tối giản"],
    },
    "Bright Winter": {
        "tops": ["Trắng tinh", "cobalt sáng", "đen contrast cao"],
        "bottoms": ["Than chì đậm", "đen", "navy sắc nét"],
        "accessories": ["Bạc bóng", "sneaker trắng sắc nét", "đồng hồ thép sáng"],
    },
}


# ═══════════════════════════════════════════════
# 12 SUB-SEASONS — Dữ liệu cứng, kiểm chứng bởi color theory
# ═══════════════════════════════════════════════

SEASON_DATA: dict[str, SeasonEntry] = {
    # ─── SPRING (Warm + Light) ─────────────────
    "Light Spring": {
        "primary": "Spring",
        "palette": ["#FAD6A5", "#F4A460", "#FFB347", "#FFDAB9", "#E8CCD7", "#B5D8CC"],
        "makeup": {
            "lip": ["Hồng đào (Peach Pink)", "Cam san hô (Coral)", "Nude hồng (Pink Nude)"],
            "eye": ["Vàng đào (Peach Gold)", "Hồng phấn (Soft Pink)", "Nâu nhạt (Light Taupe)"],
            "hair": ["Vàng mật ong (Honey Blonde)", "Nâu vàng (Golden Brown)", "Nâu caramel (Caramel)"],
        },
        "gradient_from": "#FFD6A5",
        "gradient_to": "#FFAFCC",
    },
    "True Spring": {
        "primary": "Spring",
        "palette": ["#FF6347", "#FFD700", "#FF8C00", "#98FB98", "#00CED1", "#F0E68C"],
        "makeup": {
            "lip": ["Cam tươi (Bright Coral)", "Đỏ cam (Orange Red)", "Hồng ấm (Warm Pink)"],
            "eye": ["Vàng đồng (Gold)", "Xanh ngọc (Teal)", "Cam ấm (Warm Orange)"],
            "hair": ["Vàng sáng (Light Golden)", "Đỏ đồng (Copper Red)", "Nâu vàng ấm (Warm Golden Brown)"],
        },
        "gradient_from": "#FFD6A5",
        "gradient_to": "#FFAFCC",
    },
    "Bright Spring": {
        "primary": "Spring",
        "palette": ["#FF4500", "#FFD700", "#00FF7F", "#FF69B4", "#4169E1", "#FF6347"],
        "makeup": {
            "lip": ["Đỏ tươi (Bright Red)", "Hồng fuchsia (Fuchsia)", "Cam sáng (Bright Orange)"],
            "eye": ["Xanh cobalt (Cobalt Blue)", "Xanh lá sáng (Bright Green)", "Tím sáng (Bright Purple)"],
            "hair": ["Đỏ sáng (Bright Red)", "Nâu đậm ấm (Rich Warm Brown)", "Đen xanh (Blue Black)"],
        },
        "gradient_from": "#FFD6A5",
        "gradient_to": "#FFAFCC",
    },

    # ─── SUMMER (Cool + Light) ─────────────────
    "Light Summer": {
        "primary": "Summer",
        "palette": ["#B0C4DE", "#DDA0DD", "#E6E6FA", "#FFB6C1", "#AFEEEE", "#D3D3D3"],
        "makeup": {
            "lip": ["Hồng nhạt (Soft Rose)", "Hồng tím (Mauve Pink)", "Nude lạnh (Cool Nude)"],
            "eye": ["Tím lavender (Lavender)", "Xám bạc (Silver Grey)", "Hồng pastel (Pastel Pink)"],
            "hair": ["Nâu tro (Ash Brown)", "Vàng tro (Ash Blonde)", "Nâu nhạt lạnh (Cool Light Brown)"],
        },
        "gradient_from": "#BDE0FE",
        "gradient_to": "#CDB4DB",
    },
    "True Summer": {
        "primary": "Summer",
        "palette": ["#6B8E9B", "#C08081", "#8FBC8F", "#B0A4C4", "#A9A9A9", "#D2B48C"],
        "makeup": {
            "lip": ["Hồng berry (Berry Pink)", "Mận nhạt (Soft Plum)", "Hồng cổ điển (Classic Rose)"],
            "eye": ["Xám khói (Smoky Grey)", "Tím phấn (Dusty Purple)", "Xanh thép (Steel Blue)"],
            "hair": ["Nâu tro đậm (Dark Ash Brown)", "Nâu lạnh (Cool Brown)", "Xám bạch kim (Platinum)"],
        },
        "gradient_from": "#BDE0FE",
        "gradient_to": "#CDB4DB",
    },
    "Soft Summer": {
        "primary": "Summer",
        "palette": ["#8B8682", "#B5A895", "#C4A882", "#9CAF88", "#8E8E8E", "#B6A6A0"],
        "makeup": {
            "lip": ["Hồng nude (Nude Rose)", "Nâu hồng (Rosy Brown)", "Mận nhạt (Light Mauve)"],
            "eye": ["Nâu tro (Taupe)", "Xanh rêu nhạt (Soft Sage)", "Xám ấm (Warm Grey)"],
            "hair": ["Nâu tro medium (Medium Ash Brown)", "Nâu cát (Sandy Brown)", "Bronde tro (Ash Bronde)"],
        },
        "gradient_from": "#BDE0FE",
        "gradient_to": "#CDB4DB",
    },

    # ─── AUTUMN (Warm + Deep) ──────────────────
    "Soft Autumn": {
        "primary": "Autumn",
        "palette": ["#8B7355", "#A0785A", "#C4A882", "#8B8B4B", "#D2B48C", "#BDB76B"],
        "makeup": {
            "lip": ["Nâu đất nhạt (Light Terracotta)", "Nude ấm (Warm Nude)", "Hồng đào trầm (Muted Peach)"],
            "eye": ["Nâu ấm (Warm Brown)", "Xanh olive nhạt (Soft Olive)", "Đồng nhạt (Light Bronze)"],
            "hair": ["Nâu caramel (Caramel)", "Nâu hạt dẻ nhạt (Light Chestnut)", "Nâu mật ong (Honey Brown)"],
        },
        "gradient_from": "#E9C46A",
        "gradient_to": "#8B4513",
    },
    "True Autumn": {
        "primary": "Autumn",
        "palette": ["#8B4513", "#D2691E", "#A0522D", "#CD853F", "#DEB887", "#F4A460"],
        "makeup": {
            "lip": ["Đỏ gạch (Brick Red)", "Nâu đất (Terracotta)", "Nude ấm caramel (Warm Caramel Nude)"],
            "eye": ["Đồng ánh kim (Bronze Shimmer)", "Cam cháy (Burnt Orange)", "Nâu ấm (Warm Brown)"],
            "hair": ["Nâu hạt dẻ (Chestnut)", "Nâu đỏ (Auburn)", "Nâu mật ong đậm (Dark Honey)"],
        },
        "gradient_from": "#E9C46A",
        "gradient_to": "#8B4513",
    },
    "Deep Autumn": {
        "primary": "Autumn",
        "palette": ["#4A2C2A", "#8B0000", "#2F4F2F", "#8B6914", "#4A4A2A", "#5C3317"],
        "makeup": {
            "lip": ["Đỏ rượu (Wine Red)", "Nâu sô-cô-la (Chocolate Brown)", "Berry đậm (Deep Berry)"],
            "eye": ["Xanh rừng (Forest Green)", "Đồng đậm (Deep Bronze)", "Nâu đen (Espresso)"],
            "hair": ["Nâu đậm ấm (Dark Warm Brown)", "Đỏ mahogany (Mahogany Red)", "Đen nâu (Dark Chocolate)"],
        },
        "gradient_from": "#E9C46A",
        "gradient_to": "#8B4513",
    },

    # ─── WINTER (Cool + Deep) ──────────────────
    "Deep Winter": {
        "primary": "Winter",
        "palette": ["#000080", "#8B0000", "#191970", "#2F4F4F", "#4B0082", "#006400"],
        "makeup": {
            "lip": ["Đỏ đậm (Deep Red)", "Mận (Plum)", "Berry tối (Dark Berry)"],
            "eye": ["Xanh navy (Navy Blue)", "Tím đậm (Deep Purple)", "Xám than (Charcoal)"],
            "hair": ["Đen tuyền (Jet Black)", "Nâu đen (Darkest Brown)", "Đỏ rượu vang (Burgundy)"],
        },
        "gradient_from": "#1B4F72",
        "gradient_to": "#4A235A",
    },
    "True Winter": {
        "primary": "Winter",
        "palette": ["#FF0000", "#0000FF", "#FFFFFF", "#000000", "#FF00FF", "#00FFFF"],
        "makeup": {
            "lip": ["Đỏ tươi thuần (True Red)", "Hồng fuchsia (Fuchsia)", "Berry sáng (Bright Berry)"],
            "eye": ["Bạc (Silver)", "Xanh dương sáng (Bright Blue)", "Đen (Black)"],
            "hair": ["Đen tuyền (Jet Black)", "Đen xanh (Blue Black)", "Bạch kim (Platinum Blonde)"],
        },
        "gradient_from": "#1B4F72",
        "gradient_to": "#4A235A",
    },
    "Bright Winter": {
        "primary": "Winter",
        "palette": ["#FF1493", "#00BFFF", "#7FFF00", "#FFD700", "#FF4500", "#9400D3"],
        "makeup": {
            "lip": ["Hồng neon (Neon Pink)", "Đỏ cherry (Cherry Red)", "Cam đỏ sáng (Bright Red-Orange)"],
            "eye": ["Xanh điện (Electric Blue)", "Tím sáng (Bright Violet)", "Xanh ngọc (Emerald)"],
            "hair": ["Đen bóng (Glossy Black)", "Nâu đen lạnh (Cool Darkest Brown)", "Bạch kim băng (Icy Platinum)"],
        },
        "gradient_from": "#1B4F72",
        "gradient_to": "#4A235A",
    },
}


# ═══════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════

def get_palette_override(sub_season: str) -> list[str]:
    """Lấy palette chuẩn theo sub_season — override kết quả Gemini.

    Tính năng #20: Gemini KHÔNG tự sinh hex — server thay bằng bảng cứng.

    Args:
        sub_season: Tên sub-season từ Gemini response.

    Returns:
        list[str]: 6 mã hex chuẩn.

    Raises:
        KeyError: Nếu sub_season không hợp lệ (Gemini trả sai tên).
    """
    entry = SEASON_DATA.get(sub_season)
    if entry is None:
        # Fallback: tìm gần đúng (VD: "soft autumn" → "Soft Autumn")
        for key in SEASON_DATA:
            if key.lower() == sub_season.lower():
                return SEASON_DATA[key]["palette"]
        raise KeyError(f"Sub-season không hợp lệ: {sub_season}")
    return entry["palette"]


def canonicalize_sub_season(sub_season: str) -> str | None:
    """Chuẩn hóa sub-season Gemini trả về về đúng key trong lookup table."""
    normalized = (sub_season or "").strip()
    if not normalized:
        return None

    if normalized in SEASON_DATA:
        return normalized

    lowered = normalized.casefold()
    for key in SEASON_DATA:
        if key.casefold() == lowered:
            return key

    return None


def get_season_entry(sub_season: str) -> SeasonEntry:
    """Lấy full metadata của một sub-season sau khi đã canonicalize."""
    canonical = canonicalize_sub_season(sub_season)
    if canonical is None:
        raise KeyError(f"Sub-season không hợp lệ: {sub_season}")
    return SEASON_DATA[canonical]


def get_season_gradient(sub_season: str) -> tuple[str, str]:
    """Lấy gradient colors cho SeasonGradientCard (frontend).

    Returns:
        tuple[str, str]: (gradient_from, gradient_to) hex codes.
    """
    entry = SEASON_DATA.get(sub_season)
    if entry is None:
        return ("#FAFAFA", "#E0E0E0")  # fallback neutral
    return entry["gradient_from"], entry["gradient_to"]


def get_all_sub_seasons() -> list[str]:
    """Danh sách 12 sub-season — dùng validate Gemini response."""
    return list(SEASON_DATA.keys())


def get_menswear_recommendation(sub_season: str) -> MenswearData:
    """Male-focused color application keyed by sub-season for better precision."""
    canonical = canonicalize_sub_season(sub_season)
    if canonical is None:
        raise KeyError(f"Sub-season không hợp lệ: {sub_season}")
    return MENSWEAR_BY_SUBSEASON[canonical]
