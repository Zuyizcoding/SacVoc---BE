"""Prompt bổ sung khi ảnh chưa tốt hoặc AI trả thiếu field."""

import json

from .photo_types import PhotoQualityReport


def _build_color_quality_hint_prompt(quality_flags: PhotoQualityReport) -> str:
    reasons: list[str] = []
    if quality_flags.soft_far_subject_for_color:
        reasons.append("chủ thể trong ảnh chiếm khung hình chưa nhiều")
    if quality_flags.soft_low_detail:
        reasons.append("ảnh hơi mềm hoặc thiếu chi tiết")
    if quality_flags.soft_bad_lighting:
        reasons.append("ánh sáng chưa thật sự tối ưu")
    if quality_flags.soft_color_cast:
        reasons.append("ảnh có thể bị lệch màu nhẹ")

    joined = ", ".join(reasons) if reasons else "ảnh chưa thật sự tối ưu"
    return (
        "<quality_hint>\n"
        f"Ảnh này vẫn có thể phân tích, nhưng {joined}.\n"
        "Hãy ưu tiên phân tích best-effort thay vì từ chối.\n"
        "Nếu độ chắc chắn không cao, hãy hạ confidence ở mức hợp lý: lỗi nhẹ khoảng 0.55-0.75,\n"
        "nhiều vấn đề hơn thì có thể thấp hơn, và nói rõ rằng đây là kết quả tham khảo.\n"
        "Không được từ chối chỉ vì ảnh hơi xa, hơi mềm hoặc ánh sáng chưa hoàn hảo nếu vẫn thấy rõ 1 người.\n"
        "</quality_hint>"
    )


def _build_color_completion_prompt(
    missing_fields: list[str],
    partial_result: dict,
    quality_flags: PhotoQualityReport,
) -> str:
    missing_text = ", ".join(missing_fields)
    return (
        "<task>\n"
        "Bạn vừa phân tích seasonal color từ ảnh này nhưng còn thiếu một số field quan trọng.\n"
        f"Hãy phân tích lại cùng ảnh và chỉ tập trung bổ sung: {missing_text}.\n"
        "Giữ nguyên hướng phân tích đã có nếu sub_season hiện tại vẫn hợp lý.\n"
        "Nếu ảnh chưa đủ hoàn hảo, vẫn trả best-effort với confidence thấp thay vì từ chối.\n"
        "secondary_season có thể để trống nếu không chắc, nhưng sub_season phải là 1 trong 12 mùa chuẩn.\n"
        f"Partial result trước đó để tham chiếu: {json.dumps(partial_result, ensure_ascii=False)}\n"
        f"{_build_color_quality_hint_prompt(quality_flags)}\n"
        "</task>"
    )


def _build_body_completion_prompt(
    missing_fields: list[str], partial_result: dict
) -> str:
    """Prompt ngắn để yêu cầu Gemini chỉ hoàn thiện phần còn thiếu."""
    missing_text = ", ".join(missing_fields)
    return (
        "<task>\n"
        "Bạn vừa phân tích body shape từ ảnh này nhưng còn thiếu một số field bắt buộc.\n"
        f"Hãy phân tích lại cùng ảnh và trả JSON hoàn chỉnh, tập trung bổ sung: {missing_text}.\n"
        "Chỉ điền measurements khi thực sự quan sát được tỷ lệ vai/hông/eo từ ảnh;\n"
        "không được suy diễn measurements chỉ từ shape name.\n"
        "Nếu ảnh không đủ rõ để kết luận chắc chắn, ưu tiên vẫn trả kết quả với confidence thấp,\n"
        "chỉ trả JSON lỗi CLOTHING_TOO_WIDE khi trang phục che dáng quá nặng và không thể suy luận tỷ lệ cơ bản.\n"
        "Không được bịa mẹo mặc đồ chung chung hoặc cố làm kết quả trông chắc chắn hơn thực tế.\n"
        f"Partial result trước đó để tham chiếu: {json.dumps(partial_result, ensure_ascii=False)}\n"
        "</task>"
    )


def _build_body_best_effort_prompt() -> str:
    """Prompt fail-safe: force best-effort analysis for borderline wide clothing."""
    return (
        "<task>\n"
        "Ảnh này có thể mặc đồ hơi rộng, nhưng hãy ưu tiên phân tích best-effort.\n"
        "Yêu cầu:\n"
        "1) VẪN trả đầy đủ JSON body shape (shape, confidence, measurements, styling_tips, avoid, explanation).\n"
        "2) Đặt confidence thấp (0.25-0.45) nếu độ chắc chắn không cao.\n"
        "3) Trong explanation phải nêu rõ là trang phục hơi rộng nên kết quả chỉ tham khảo.\n"
        "4) CHỈ trả lỗi CLOTHING_TOO_WIDE khi bị che dáng cực nặng: không nhìn thấy tương quan vai/eo/hông một cách tối thiểu.\n"
        "5) Không bịa dữ liệu, không dùng câu chung chung.\n"
        "</task>"
    )


def _build_body_medium_quality_hint_prompt(quality_flags: PhotoQualityReport) -> str:
    detail_score = quality_flags.detail_score
    return (
        "<quality_hint>\n"
        "Ảnh này là ảnh toàn thân mức trung bình: vẫn nhìn thấy silhouette tổng thể, "
        "nhưng chi tiết không quá cao.\n"
        f"Detail score tham khảo: {detail_score}.\n"
        "Hãy ưu tiên phân tích best-effort dựa trên tương quan vai/eo/hông và form tổng thể.\n"
        "Không được từ chối chỉ vì khuôn mặt nhỏ, ảnh hơi mềm hoặc áo thun hơi rộng.\n"
        "Chỉ trả CLOTHING_TOO_WIDE khi thực sự không nhìn thấy tương quan vai/eo/hông ở mức tối thiểu.\n"
        "Nếu độ chắc chắn thấp, hạ confidence xuống 0.25-0.45 và ghi rõ là kết quả tham khảo.\n"
        "</quality_hint>"
    )
