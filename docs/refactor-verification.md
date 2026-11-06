# Nghiệm thu refactor

Kiểm tra ngày 06/11/2026. Các thay đổi nằm trong hai working tree, chưa tạo commit hoặc triển khai server.

## Kết quả kiểm tra

| Kiểm tra | Kết quả |
|---|---|
| Backend `pytest` | 62 tests đạt, gồm các hành vi cũ và regression tests mới. |
| Ruff 0.9.4 | `check` đạt với E4/E7/E9/F/I; `format --check` đạt. |
| Flutter analyzer | Không có issues. |
| Dart formatter | 47 file trong `lib`/`test` đã đúng format. |
| Flutter regression tests | 17 tests đạt. |
| Build Android debug | Thành công với Flutter 3.41.5, Dart 3.11.3, JDK 17. |

APK local: `SacVoc---FE/build/app/outputs/flutter-apk/app-debug.apk` (154.294.219 bytes). SDK Flutter/Ruff được chuẩn bị trong thư mục tạm; không nâng phiên bản dependency của project. Build lần đầu tự cài các thành phần Android SDK/NDK/CMake còn thiếu.

## Bảo toàn hành vi

- Đối chiếu snapshot trước refactor với 30 kịch bản F1/F2: HTTP status, response, nội dung prompt và thứ tự gọi AI khớp chính xác.
- Đối chiếu 480 tổ hợp F3: điểm, reason tags, thứ tự và lựa chọn đa dạng khớp chính xác.
- 240 query templates khớp file vectors và signature `9c04976c9db0fc3619050c0da9820c187d38c7d6314606f217ca94e9db191fb4`.
- Catalog CSV, query vectors và hai manifest giữ nguyên bytes; prompt constants và season lookup giữ nguyên giá trị.
- `pubspec.yaml` giữ nguyên so với snapshot; `pubspec.lock` giữ nguyên so với HEAD. Các registrant do Flutter tự sinh không được đưa vào refactor.
- Kiểm tra SQLite trong bộ nhớ với vectors precomputed thật xác nhận recommendation/readiness hoạt động mà không tải encoder runtime.

## Đối chiếu giao diện

Home, F1, F2 và history được render từ cả source trước và sau refactor, với cùng dữ liệu live/demo. Kiểm tra đầu và cuối màn hình trên viewport 800×600 và 432×936: tổng cộng 32 cặp ảnh khớp pixel. Font Poppins đi kèm project và MaterialIcons được nạp trong bộ đối chiếu.

Các ảnh và script QA một lần nằm ở `SacVoc---FE/build/refactor_qa/`, được Git ignore. Regression tests dùng chung được lưu trong `test/`. Đây là kiểm tra bằng Flutter test renderer, không phải nghiệm thu trên thiết bị Android thật.

## Điểm bắt đầu đọc sau refactor

| File điều phối | Trước | Sau |
|---|---:|---:|
| Backend endpoint phân tích | 1.020 dòng | 122 dòng |
| Gemini client | 521 dòng | 163 dòng |
| Outfit service | 797 dòng | 169 dòng |
| Flutter router | 373 dòng | 100 dòng |

Số dòng gồm comments/dòng trống; mục tiêu là trách nhiệm rõ và luồng dễ lần theo. Phần triển khai được chuyển sang module có tên cụ thể, không bị xóa để rút ngắn file.

## Giới hạn của kiểm chứng

AI được giả lập trong tests; không gọi Gemini hoặc tải FashionCLIP, không seed/prune DB người dùng. Việc bảo toàn hành vi không phải đánh giá lại độ chính xác AI hay chất lượng tư vấn thời trang. Không có migration dữ liệu mới, thay đổi định danh thiết bị, đổi thiết kế UI hoặc thay đổi API wire format.

Đọc tiếp [kiến trúc](architecture.md), [hướng dẫn backend](development.md) và [onboarding Flutter](../../SacVoc---FE/docs/onboarding.md).
