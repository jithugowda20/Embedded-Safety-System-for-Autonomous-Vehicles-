import cv2
import numpy as np

# Paths to YOLO model files (update with your file paths)
weights_path = 'D:/projects/yolov3.weights'
config_path = 'D:/projects/yolov3.cfg'
names_path = 'D:/projects/coco.names'

# Load the YOLO model
net = cv2.dnn.readNet(weights_path, config_path)

# Load COCO class labels
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture from the default camera
cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide a video file path

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names with compatibility for different OpenCV versions
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # Forward pass
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Process the outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maxima Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels if detections are found
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)  # Set color for bounding box (green)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the frame with detections
    cv2.imshow("YOLO Object Detection", frame)

    # Press 'q' to break out of the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
