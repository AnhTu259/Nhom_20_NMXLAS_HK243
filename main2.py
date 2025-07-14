import cv2
import numpy as np
from ultralytics import solutions

cap = cv2.VideoCapture("duongpho.mp4")
assert cap.isOpened(), "Error reading video file"

# region_points = [(20, 400), (1080, 400)]                                      # line counting
region_points = [[1877, 568], [40, 568], [40, 943], [1881, 943]]  # rectangle region
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # polygon region

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize object counter object
counter = solutions.ObjectCounter(
    show=True,  # display the output
    region=region_points,  # pass region points
    model="yolo11n.pt",  # model="yolo11n-obb.pt" for object counting with OBB model.
    # classes=[0, 2],  # count specific classes i.e. person and car with COCO pretrained model.
    tracker="bytetrack.yaml",
    conf=0.4,
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    im0_processed = cv2.GaussianBlur(im0, (5, 5), 0)  # Lọc nhiễu
    im0_processed = cv2.convertScaleAbs(im0_processed, alpha=1.2, beta=10)  # Tăng tương phản và độ sáng
    # Chuyển sang HSV và điều chỉnh (tùy chọn)
    hsv = cv2.cvtColor(im0_processed, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.1, 0, 255)  # Tăng độ sáng (Value)
    im0_processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    results = counter(im0_processed)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()   
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows