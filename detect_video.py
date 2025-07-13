import torch
import cv2
import os

# Load model
model = torch.hub.load('yolov5', 'yolov5s', source='local')
model.conf = 0.25  # confidence threshold thấp hơn để phát hiện vật thể xa
model.iou = 0.45

# Load video
cap = cv2.VideoCapture('input/video/highway.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/video/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    results.render()
    output_frame = results.ims[0].copy()

    labels = results.pandas().xyxy[0]['name']
    counts = labels.value_counts()
    text = ', '.join([f"{cls}: {cnt}" for cls, cnt in counts.items()])
    cv2.putText(output_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    out.write(output_frame)
    cv2.imshow('Video Detection', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()