# Data Directory

Thư mục này chứa dữ liệu cho training và testing.

## Cấu trúc

```
data/
├── videos/          # Video files for testing
├── images/          # Image files
├── datasets/        # Training datasets
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/     # Annotation files
```

## Dataset cho Training Stroke Detection

Để train model phát hiện đột quỵ, bạn cần:

1. **Video clips** của các triệu chứng đột quỵ:
   - Face drooping (méo mặt)
   - Arm weakness (yếu tay)
   - Balance loss (mất thăng bằng)
   - Sudden fall (ngã đột ngột)
   - Abnormal posture (tư thế bất thường)

2. **Annotations** cho mỗi video:
   - Frame-level labels
   - Skeleton keypoints (extracted by pose estimator)
   - Symptom class

3. **Format**:
   ```
   datasets/
   ├── train/
   │   ├── videos/
   │   │   ├── stroke_001.mp4
   │   │   └── ...
   │   └── annotations/
   │       ├── stroke_001.json
   │       └── ...
   └── val/
       └── ...
   ```

## Nguồn dữ liệu

- Medical simulation videos
- Clinical recordings (với consent)
- Synthetic data generation
- Public datasets (nếu có)

**Lưu ý**: Dữ liệu y tế cần tuân thủ quy định về bảo mật và privacy.
