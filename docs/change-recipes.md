# Ba thay đổi thường gặp

## 1. Thêm một field kết quả

Ví dụ cần thêm nhãn mô tả vào kết quả màu:

1. Xác định field là suy luận từ AI hay dữ liệu backend tự bổ sung.
2. Nếu AI cung cấp: cập nhật schema suy luận, prompt và kiểm tra thiếu field trong workflow. Nếu backend tính được: bổ sung tại bước chuẩn hóa kết quả.
3. Thêm field vào schema API; dữ liệu history cũ cần giá trị mặc định hoặc cho phép null.
4. JSON đầy đủ đã được CRUD lưu. Chỉ cần migration khi bổ sung cột ORM riêng để truy vấn, không phải với mọi field JSON mới.
5. Cập nhật model Flutter và view kết quả dùng chung. Kiểm tra cả màn phân tích lẫn history.
6. Thêm tests cho dữ liệu mới, history cũ thiếu field và demo cache. Việc thêm field công khai là thay đổi API cần được đánh giá riêng.

## 2. Chỉnh nội dung hiển thị

Đi đến `ColorResultView`, `BodyResultView`, `PhotoQualityCard` hoặc `AnalysisSourceBanner` ở frontend. Nội dung từ AI/lookup cần sửa tại nguồn backend; nội dung cố định của giao diện sửa tại widget. Chạy widget tests với cả `live` và `degraded_demo` để chắc rằng cảnh báo nguồn kết quả vẫn hiện đúng.

## 3. Chỉnh một quy tắc xếp hạng

1. Bắt đầu từ `recommendation/ranking.py`: xác định đang đổi similarity, bonus mùa/dáng, hệ số confidence hay chọn đa dạng.
2. Thêm ví dụ trước/sau với vectors cố định: sản phẩm nào phải tăng hạng và trường hợp nào phải giữ nguyên.
3. Sửa quy tắc/trọng số và chạy `tests/test_outfit_service.py`, `tests/test_recommendation_pipeline.py`.
4. Đổi quy tắc ranking không tự yêu cầu encode lại. Đổi chuỗi query trong `query_builder.py` thì có: cần chuẩn bị lại query vectors và manifest tương ứng.
5. Chạy benchmark offline nếu có catalog thật và nhãn đánh giá; điểm match tăng không tự chứng minh chất lượng đề xuất tốt hơn.

## Quy ước khi review

Tên hàm mô tả hành động; workflow đọc theo thứ tự; endpoint không chứa chính sách AI. Giữ một nơi sở hữu mỗi công thức hoặc khai báo dùng chung. Comments giải thích lý do và giới hạn. Khi sửa fallback hoặc retry, xem tests ghi nhận số lượt gọi và cách lưu dữ liệu trước khi đổi implementation.
