import cv2
import numpy as np
from ultralytics import solutions

# Khởi tạo video
cap = cv2.VideoCapture("highway.mp4")
assert cap.isOpened(), "Error reading video file"

# Xác định vùng đếm
region_points = [[0, 675], [1284, 675], [1284, 397], [912, 278], [1, 278]]  # rectangle region

# Lấy kích thước và fps
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))  # Thay mp4v bằng XVID cho ổn định

# Initialize object counter object
counter = solutions.ObjectCounter(
    show=True,  # Hiển thị output
    region=region_points,  # Vùng đếm
    model="yolov5s.pt",  # Thay đổi thành YOLOv5 Small
    # classes=[0, 2],  # Tùy chọn: chỉ đếm person và car
    tracker="bytetrack.yaml",
    conf=0.4,
)

# Process video
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

    # print(results)  # In chi tiết để kiểm tra (bật lên nếu cần)
    video_writer.write(results.plot_im)  # write the processed frame.
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()