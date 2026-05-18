# 🧠 Stroke AI - Modern Detection System

Hệ thống nhận diện đột quỵ và té ngã hiện đại sử dụng công nghệ AI tiên tiến, được thiết kế cho các ứng dụng IoT và giám sát y tế.

## ✨ Tính năng chính
- **AI Nhận diện**: Sử dụng YOLOv8 (Person Detection) và MediaPipe (Pose Estimation).
- **Phát hiện sự cố**: Tự động nhận diện các triệu chứng như Ngã đột ngột, Tư thế bất thường, và Mất thăng bằng.
- **GUI Hiện đại**: Dashboard cao cấp xây dựng bằng CustomTkinter.
- **Cloud Sync**: Tự động tải ảnh cảnh báo lên Supabase và lưu trữ lịch sử sự cố.
- **Đa nguồn**: Hỗ trợ Webcam và Video file.

## 📁 Cấu trúc thư mục
```
Stroke_al/
├── app/                # Mã nguồn chính
│   ├── ai/             # Engine xử lý AI (YOLOv8, MediaPipe)
│   ├── cloud/          # Kết nối Supabase
│   ├── gui/            # Giao diện người dùng
│   └── utils/          # Công cụ hỗ trợ và Visualization
├── assets/             # Tài nguyên (Icons, Models)
├── data/               # Dữ liệu tạm thời
├── .env                # Cấu hình Supabase (URL, KEY)
├── main.py             # File chạy chính
└── requirements.txt    # Danh sách thư viện
```

## 🚀 Hướng dẫn cài đặt

1. **Cài đặt thư viện**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Cấu hình Supabase**:
   Mở file `.env` và điền thông tin kết nối Supabase của bạn:
   ```env
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-api-key
   SUPABASE_BUCKET=stroke-detections
   ```

3. **Chạy ứng dụng**:
   ```bash
   python main.py
   ```

## 🛠 Công nghệ sử dụng
- **Python 3.10+**
- **Ultralytics YOLOv8**: Nhận diện người chính xác cao.
- **MediaPipe**: Theo dõi khung xương thời gian thực.
- **CustomTkinter**: Giao diện người dùng hiện đại (Premium Look).
- **Supabase**: Backend-as-a-Service cho lưu trữ ảnh và data.

---
*Phát triển bởi Đội ngũ Chuyên gia AI - IoT*
