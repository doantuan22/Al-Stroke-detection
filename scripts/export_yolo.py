import os
import sys
import torch

def export_model():
    print("=" * 60)
    print("      YOLOv8 to TensorRT Engine Export Tool")
    print("=" * 60)

    # 1. Kiểm tra CUDA
    if not torch.cuda.is_available():
        print("[-] LỖI: Không tìm thấy GPU CUDA trên hệ thống của bạn.")
        print("    TensorRT yêu cầu GPU NVIDIA có hỗ trợ CUDA.")
        sys.exit(1)

    print(f"[+] Tìm thấy GPU: {torch.cuda.get_device_name(0)}")
    print(f"[+] Phiên bản CUDA (PyTorch): {torch.version.cuda}")

    # 2. Nhập file model
    model_path = input("\nNhập đường dẫn file model YOLO (mặc định: yolov8n.pt): ").strip()
    if not model_path:
        model_path = "yolov8n.pt"

    if not os.path.exists(model_path):
        print(f"[-] LỖI: Không tìm thấy file model '{model_path}'")
        sys.exit(1)

    # 3. Chọn kiểu định dạng export
    print("\nChọn kiểu TensorRT engine:")
    print("1. FP16 (Khuyên dùng - Nhanh, chính xác, ổn định)")
    print("2. INT8  (Cực nhanh, yêu cầu dữ liệu calibration - Phức tạp)")
    choice = input("Lựa chọn (1 hoặc 2, mặc định 1): ").strip()
    
    half_mode = True
    int8_mode = False
    
    if choice == "2":
        half_mode = False
        int8_mode = True
        print("[!] Lưu ý: Chế độ INT8 yêu cầu dataset calibration để đảm bảo độ chính xác.")

    print("\n[+] Đang tải model YOLOv8...")
    from ultralytics import YOLO
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[-] LỖI khi tải model: {e}")
        sys.exit(1)

    print(f"[+] Bắt đầu export model {model_path} sang TensorRT (.engine)...")
    print("[!] Quá trình này có thể mất từ 5 đến 15 phút, vui lòng KHÔNG tắt cửa sổ lệnh.")
    
    try:
        # Thực hiện export qua ultralytics API
        # TensorRT export tự động chạy trên GPU
        export_path = model.export(
            format="engine",
            half=half_mode,
            int8=int8_mode,
            dynamic=False,
            simplify=True,
            workspace=4.0, # 4GB workspace memory cho TensorRT compile
            verbose=True
        )
        print("=" * 60)
        print(f"[+] THÀNH CÔNG! File TensorRT đã được lưu tại:")
        print(f"    {export_path}")
        print("[+] Airport Security AI sẽ tự động sử dụng file .engine này khi chạy chế độ CUDA.")
        print("=" * 60)
    except Exception as e:
        print("\n[-] LỖI khi export sang TensorRT:")
        print(f"    {e}")
        print("\n[👉] Hướng dẫn khắc phục lỗi thường gặp:")
        print("1. Đảm bảo bạn đã cài đặt thư viện 'tensorrt' thông qua pip:")
        print("   pip install tensorrt")
        print("2. Đảm bảo NVIDIA CUDA Toolkit và cuDNN đã được cài đặt và thêm vào biến môi trường PATH.")
        print("3. Phiên bản driver GPU NVIDIA của bạn phải tương thích với CUDA Toolkit được sử dụng.")
        print("4. Để đơn giản, bạn có thể tham khảo tài liệu của Ultralytics tại: https://docs.ultralytics.com/modes/export/")
        print("=" * 60)

if __name__ == "__main__":
    export_model()
