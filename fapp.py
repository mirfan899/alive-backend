from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import numpy as np
import io
import os

app = FastAPI()

# Directories to store reference images and videos for feature matching
REFERENCE_IMAGES_DIR = "images"
REFERENCE_VIDEOS_DIR = "videos"
os.makedirs(REFERENCE_IMAGES_DIR, exist_ok=True)
os.makedirs(REFERENCE_VIDEOS_DIR, exist_ok=True)

# Load your feature database
# Store both features and corresponding video paths
db_features = []


def extract_features(image):
    # Use ORB for feature detection (you can switch to Kornia or LightGlue)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def load_reference_data():
    global db_features
    db_features = []
    for filename in os.listdir(REFERENCE_IMAGES_DIR):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(REFERENCE_IMAGES_DIR, filename)
            video_path = os.path.join(REFERENCE_VIDEOS_DIR, os.path.splitext(filename)[0] + ".mp4")
            if os.path.exists(video_path):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                keypoints, descriptors = extract_features(image)
                db_features.append((keypoints, descriptors, video_path))


# Load existing reference images and videos on startup
load_reference_data()


@app.post("/upload_reference/")
async def upload_reference_image(file: UploadFile = File(...), video: UploadFile = File(...)):
    image_path = os.path.join(REFERENCE_IMAGES_DIR, file.filename)
    video_path = os.path.join(REFERENCE_VIDEOS_DIR, os.path.splitext(file.filename)[0] + ".mp4")

    with open(image_path, "wb") as img_buffer:
        img_buffer.write(await file.read())

    with open(video_path, "wb") as vid_buffer:
        vid_buffer.write(await video.read())

    load_reference_data()
    return JSONResponse(content={"message": "Reference image and video uploaded, features extracted successfully."})


def match_features(descriptors, db_descriptors):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors, db_descriptors)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


@app.get("/video")
async def video_feed():
    def generate():
        cap = cv2.VideoCapture(0)

        # Load YOLOv4-tiny for mobile phone detection
        net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
        layer_names = net.getLayerNames()

        # Fix the output layer extraction to handle scalar outputs
        try:
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
        except AttributeError:
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        video_caps = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            height, width, channels = frame.shape

            # Prepare the frame for YOLO
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5 and classes[class_id] == "cell phone":
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    roi_color = frame[y:y + h, x:x + w]
                    keypoints, descriptors = extract_features(roi_color)

                    if db_features:
                        for db_keypoints, db_descriptors, video_path in db_features:
                            matches = match_features(descriptors, db_descriptors)
                            if matches and len(matches) > 10:  # Ensure sufficient matches for better accuracy
                                if video_path not in video_caps:
                                    video_caps[video_path] = cv2.VideoCapture(video_path)

                                video_cap = video_caps[video_path]
                                v_ret, video_frame = video_cap.read()

                                if v_ret:
                                    # Resize the video frame to fit the detected phone screen
                                    video_frame_resized = cv2.resize(video_frame, (w, h))

                                    # Adjust height if video frame doesn't fit perfectly
                                    frame_h, frame_w = frame[y:y + h, x:x + w].shape[:2]
                                    video_h, video_w = video_frame_resized.shape[:2]

                                    if video_h > frame_h:
                                        video_frame_resized = video_frame_resized[:frame_h, :]
                                    elif video_h < frame_h:
                                        padding = frame_h - video_h
                                        video_frame_resized = cv2.copyMakeBorder(video_frame_resized, 0, padding, 0, 0,
                                                                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))

                                    # Create a mask for the phone screen area
                                    mask = frame.copy()
                                    mask[y:y + video_frame_resized.shape[0],
                                    x:x + video_frame_resized.shape[1]] = video_frame_resized

                                    # Blend the original frame with the video overlay
                                    alpha = 0.7  # Transparency factor
                                    frame = cv2.addWeighted(frame, 1 - alpha, mask, alpha, 0)
                                break

                    # Draw rectangle around detected phone
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Cell Phone", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            _, img_encoded = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')

# Run the server with: uvicorn feature_matching_api:app --reload
