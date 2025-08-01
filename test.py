import cv2
import numpy as np
import easyocr
from ultralytics import YOLO, solutions

# Hàm điều chỉnh kích thước khung hình, giữ tỷ lệ nếu cần
def resize_frame(frame, target_width, target_height, keep_aspect_ratio=True):
    h, w = frame.shape[:2]
    if keep_aspect_ratio:
        aspect_ratio = w / h
        target_ratio = target_width / target_height
        if aspect_ratio > target_ratio:
            new_w = target_width
            new_h = int(target_width / aspect_ratio)
        else:
            new_h = target_height
            new_w = int(target_height * aspect_ratio)
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        top = (target_height - new_h) // 2
        bottom = target_height - new_h - top
        left = (target_width - new_w) // 2
        right = target_width - new_w - left
        padded_frame = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return padded_frame
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

# Hàm điều chỉnh tọa độ vùng đếm
def adjust_region_points(region_points, orig_w, orig_h, target_w, target_h):
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    adjusted_points = [[int(x * scale_x), int(y * scale_y)] for x, y in region_points]
    adjusted_points = [[min(max(0, x), target_w-1), min(max(0, y), target_h-1)] for x, y in adjusted_points]
    return adjusted_points

# Kích thước mục tiêu
target_width = 1280
target_height = 720

cap = cv2.VideoCapture("xecobienso.mp4")
assert cap.isOpened(), "Không mở được video!"

# Lấy kích thước và fps
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
print(f"Kích thước gốc: Width={w}, Height={h}")

# Khởi tạo video writer
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (target_width, target_height))

# Vùng đếm
region_points_orig = [[0, 1000], [3800, 1000], [3800, 3000], [0, 3000]]
region_points_adjusted = adjust_region_points(region_points_orig, w, h, target_width, target_height)
print(f"Region points điều chỉnh: {region_points_adjusted}")

# Model đếm và nhận diện xe
counter = solutions.ObjectCounter(
    show=False,
    region=region_points_adjusted,
    model="yolov8s.pt",
    tracker="bytetrack.yaml",
    conf=0.4,
)
vehicle_model = YOLO("yolov8s.pt")

# Model nhận diện biển số và OCR
plate_model = YOLO("runs/detect/lp_detector/weights/best.pt")
ocr_reader = easyocr.Reader(['en'])

plate_by_id = {}

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Hết video hoặc lỗi.")
        break

    # Resize khung hình
    im0_display = resize_frame(im0, target_width, target_height)

    # Xử lý ảnh đầu vào (tăng độ nét cho object detector)
    im0_processed = cv2.GaussianBlur(im0, (5, 5), 0)
    im0_processed = cv2.convertScaleAbs(im0_processed, alpha=1.2, beta=10)
    hsv = cv2.cvtColor(im0_processed, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
    im0_processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    im0_processed_resized = resize_frame(im0_processed, target_width, target_height)

    # Đếm xe
    results = counter(im0_processed_resized)
    results_plot = resize_frame(results.plot_im, target_width, target_height)

    # Vẽ vùng đếm
    pts = np.array(region_points_adjusted, np.int32)
    cv2.polylines(results_plot, [pts], isClosed=True, color=(255, 0, 255), thickness=2)

    # Lấy ID xe
    vehicle_results = vehicle_model.track(im0, persist=True, tracker="bytetrack.yaml", conf=0.2)

    # Nhận diện biển số
    plate_results = plate_model(im0, conf=0.3)
    for r in plate_results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            plate_crop = im0[y1:y2, x1:x2]

            # ====== XỬ LÝ ẢNH BIỂN SỐ ======
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)

            # OCR
            ocr_out = ocr_reader.readtext(sharpened, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            if ocr_out and ocr_out[0][2] > 0.4:
                plate_text = ocr_out[0][1].upper()
                confidence = ocr_out[0][2]

                # Gán ID xe
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                vehicle_id = None
                if vehicle_results[0].boxes is not None and vehicle_results[0].boxes.id is not None:
                    for track in vehicle_results[0].boxes:
                        track_x1, track_y1, track_x2, track_y2 = map(int, track.xyxy[0])
                        if track_x1 <= center_x <= track_x2 and track_y1 <= center_y <= track_y2:
                            vehicle_id = int(track.id)
                            break

                # Cập nhật biển số theo ID
                if vehicle_id is not None:
                    if vehicle_id not in plate_by_id or confidence > plate_by_id[vehicle_id].get('confidence', 0):
                        plate_by_id[vehicle_id] = {'plate_text': plate_text, 'confidence': confidence}
                        print(f"Phát hiện biển số cho xe ID {vehicle_id}: {plate_text} (Độ tin cậy: {confidence:.2f})")

                # Vẽ lên khung hình
                scale_x = target_width / w
                scale_y = target_height / h
                x1_display, y1_display = int(x1 * scale_x), int(y1 * scale_y)
                x2_display, y2_display = int(x2 * scale_x), int(y2 * scale_y)
                display_text = f"{plate_text} (ID: {vehicle_id})" if vehicle_id is not None else plate_text
                cv2.putText(results_plot, display_text, (x1_display, y1_display - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(results_plot, (x1_display, y1_display), (x2_display, y2_display), (0, 255, 0), 2)

    # Hiển thị và ghi video
    cv2.imshow("Phát hiện phương tiện và biển số", results_plot)
    video_writer.write(results_plot)
    if cv2.waitKey(1) == ord('q'):
        break

# Xuất kết quả
print("\nDanh sách biển số xe không lặp lại (theo ID xe):")
for vehicle_id, data in sorted(plate_by_id.items()):
    print(f"Xe ID {vehicle_id}: {data['plate_text']} (Độ tin cậy: {data['confidence']:.2f})")

cap.release()
video_writer.release()
cv2.destroyAllWindows()
