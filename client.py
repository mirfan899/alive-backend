import cv2
import socketio
import numpy as np

# Connect to the server
sio = socketio.Client()
sio.connect('http://127.0.0.1:5000')

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

# Handle matched frames from the server
@sio.on('matched_frame')
def on_matched_frame(data):
    """Displays matched frames received from the server."""
    frame_data = np.frombuffer(data, np.uint8)
    matched_frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

    if matched_frame is not None:
        cv2.imshow('Matched Frame from Server', matched_frame)
    else:
        print("Failed to decode the matched frame.")

# Stream video frames to the server
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to capture frame from camera.")
        continue

    # Show the captured frame
    cv2.imshow('Captured Frame', frame)

    # Encode frame as JPEG
    sio.emit('video_stream', frame.tobytes())  # Send frame to server


    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sio.disconnect()
