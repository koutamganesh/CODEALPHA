import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model (you can use 'yolov8n.pt' for faster performance)
model = YOLO('yolov8s.pt')  # Downloaded automatically if not present

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=30)

# Use webcam or video file
cap = cv2.VideoCapture(0)  # Change to 'video.mp4' for file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, cls = result
        class_id = int(cls)
        label = model.names[class_id]
        detections.append(([x1, y1, x2 - x1, y2 - y1], score, label))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Object Detection & Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
