
# Object Counting with YOLOv8 and Region-based Tracking

## Giới thiệu

Dự án này sử dụng mô hình YOLOv8 để **nhận diện và đếm số lượng vật thể** đi qua một vùng cụ thể trong video. Ngoài ra, video đầu vào được **nâng cao chất lượng hình ảnh** để xử lý tốt hơn trong điều kiện tối hoặc mờ.

## Cấu trúc mã

- `main.py` – Tệp chính thực hiện:
  - Đọc video
  - Nâng cao chất lượng ảnh
  - Resize video
  - Nhận diện & đếm đối tượng trong vùng chỉ định
- Sử dụng `ultralytics.solutions.ObjectCounter` để đếm.

## Yêu cầu hệ thống

```bash
Python 3.11.5
OpenCV 4.2
ultralytics >= 8.0.20
numpy
```

## 🧠 Thư viện chính sử dụng

| Thư viện       | Mục đích chính                              |
|----------------|----------------------------------------------|
| `cv2 (OpenCV)` | Đọc, xử lý, hiển thị và ghi video           |
| `numpy`        | Tính toán ma trận và xử lý ảnh              |
| `ultralytics`  | Dùng YOLOv8 và module ObjectCounter          |

## 🚀 Hướng dẫn chạy

1. **Cài đặt thư viện cần thiết**:
```bash
pip install opencv-python ultralytics numpy
```

2. **Tải mô hình YOLOv8** *(nếu chưa có)*:
```bash
# Ví dụ với YOLOv8s (small)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

3. **Chạy chương trình**:
```bash
python main.py
```

## Chức năng chính

### 1. **Nâng cao ảnh đầu vào (`enhance_dark_blurry`)**:
- Sử dụng **CLAHE** để tăng độ sáng kênh V trong không gian HSV.
- Áp dụng **bộ lọc sharpen** làm nét ảnh.
- **Làm mượt vùng xa** phía trên khung hình để giảm nhiễu và vỡ hình.

### 2. **Xác định vùng đếm**:
- Dựa trên tọa độ gốc của video gốc, vùng được **scale tự động** theo tỷ lệ resize.

### 3. **Đếm vật thể với YOLOv8 + ObjectCounter**:
- Sử dụng module `ultralytics.solutions.ObjectCounter`.
- Chạy mỗi khung hình qua mô hình, kiểm tra xem vật thể có băng qua vùng đếm không.
- Kết quả vẽ được hiển thị trực tiếp và ghi vào file.

## 🧾 Tham số hàm chính

```python
count_objects_in_region(
    video_path="videomo2.mp4",               # Đường dẫn video gốc
    output_video_path="output_scaled.avi",  # Video đầu ra đã xử lý
    model_path="yolov8s.pt"                 # Trọng số mô hình YOLOv8
)
```

## Output

- Video sau xử lý: `output_scaled.avi`
- Hiển thị khung hình có đếm đối tượng trực tiếp trên cửa sổ.

## Ghi chú thêm

- Nếu video đầu vào quá tối, bạn có thể thay đổi `clipLimit` trong `CLAHE` để làm sáng hơn.
- Có thể thay đổi mô hình YOLO khác như `yolov8n.pt`, `yolov8m.pt`, tùy theo độ chính xác / tốc độ yêu cầu.
- Cửa sổ kết quả sẽ tắt khi bạn nhấn **`q`**.
