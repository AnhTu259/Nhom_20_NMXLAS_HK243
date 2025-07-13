import torch
import cv2
import os

# Load model
model = torch.hub.load('yolov5', 'yolov5s', source='local')
model.conf = 0.25  # confidence threshold, giảm để phát hiện vật thể xa
model.iou = 0.45

# Load image
image_path = 'input/img/pic2.jpg'
img = cv2.imread(image_path)

# Inference
results = model(img)
labels = results.pandas().xyxy[0]['name']
counts = labels.value_counts()

# Render results
results.render()
output_img = results.ims[0].copy()

# Overlay object count on image
text = ', '.join([f"{cls}: {cnt}" for cls, cnt in counts.items()])
cv2.putText(output_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

cv2.imwrite('output/img/output.jpg', output_img)

# Show result
cv2.imshow('Image Detection', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()