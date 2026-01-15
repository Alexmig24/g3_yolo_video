from ultralytics import YOLO
import cv2

# Modelo pequeño (rápido en CPU)
model = YOLO("yolov8n.pt")
model.conf = 0.4

# Forzar CPU
model.to("cpu")

cap = cv2.VideoCapture(0)

# Reducir resolución = más FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for r in results:
        frame = r.plot()

    cv2.imshow("YOLOv8 - CPU", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
