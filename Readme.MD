**Sử dụng yolov5s.pt để chạy video file test.py**<br>
```python
cap = cv2.VideoCapture("highway.mp4")
```
_đây là đường dẫn để mở video (thay thế highway bằng file nào đó dạng video<br>
```python
region_points = [[0, 675], [1284, 675], [1284, 397], [912, 278], [1, 278]]
```
_Đây là vùng đếm để tạo ô in, out. Mảng bên trong là vị trí tọa độ ảnh, VD: region_points[0]=[0,675] là tọa độ x=0 (chiều ngang ảnh), y=675 (chiều dọc ảnh).<br>
_Đây là đường dẫn trang hỗ trợ tìm tọa độ ảnh: [https://polygonzone.roboflow.com/](https://polygonzone.roboflow.com/)
```python
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    # Tiền xử lý khung hình
    im0_processed = cv2.GaussianBlur(im0, (5, 5), 0)  # Lọc nhiễu
    im0_processed = cv2.convertScaleAbs(im0_processed, alpha=1.2, beta=10)  # Tăng tương phản và độ sáng
    hsv = cv2.cvtColor(im0_processed, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)  # Tăng độ sáng (Value)
    im0_processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    results = counter(im0_processed)
```
_Đoạn code trên có phần tiền xử lý khung hình để thay đôi độ sáng và màu sắc ảnh cho dễ nhận diện, có thể dùng nó chung cho các file khác.
