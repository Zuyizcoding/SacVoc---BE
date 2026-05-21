# Evaluation Benchmarks

Thư mục này chứa benchmark mẫu cho 3 luồng:

- `f1_color_cases.sample.json`: benchmark gán nhãn mùa sắc
- `f2_body_cases.sample.json`: benchmark gán nhãn body shape nam
- `f3_recommend_cases.sample.json`: benchmark cho recommendation

## Cách dùng

### 1. F3 recommendation

```powershell
python -m scripts.evaluate_f3 --benchmark evals/f3_recommend_cases.sample.json
```

### 2. F1/F2 label evaluation

Chuẩn bị file predictions JSON với cấu trúc:

```json
[
  {
    "case_id": "f1-soft-autumn-001",
    "predicted_sub_season": "Soft Autumn",
    "predicted_primary_season": "Autumn"
  }
]
```

Hoặc cho F2:

```json
[
  {
    "case_id": "f2-rectangle-001",
    "predicted_shape": "Rectangle"
  }
]
```

Chạy:

```powershell
python -m scripts.evaluate_analysis_exports --benchmark evals/f1_color_cases.sample.json --predictions results/f1_predictions.json --task f1
python -m scripts.evaluate_analysis_exports --benchmark evals/f2_body_cases.sample.json --predictions results/f2_predictions.json --task f2
```

## Lưu ý

- File `.sample.json` chỉ là mẫu format.
- Khi bắt đầu đánh giá thật, hãy nhân bản thành file benchmark riêng và điền nhãn người chấm.
- Nên khóa benchmark cố định để so được giữa các vòng tune prompt, data, và rerank.
