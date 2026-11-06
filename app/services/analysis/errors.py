"""Các lỗi nghiệp vụ được endpoint chuyển thành phản hồi HTTP cũ."""


class AnalysisError(Exception):
    def __init__(self, message: str, error_code: str):
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class PhotoValidationError(AnalysisError):
    """Ảnh không đọc được hoặc quá nhỏ (HTTP 422)."""


class AnalysisRejectedError(AnalysisError):
    """AI từ chối ảnh hoặc không đủ chi tiết (HTTP 200, success=false)."""


class AnalysisUnavailableError(AnalysisError):
    """AI trả dữ liệu không hợp lệ hoặc hệ thống lỗi (HTTP 503)."""


class GeminiOverloadedError(Exception):
    """Upstream quá tải; workflow sẽ dùng dữ liệu demo."""
