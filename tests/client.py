import threading

import cv2
import base64
import numpy as np
import socketio  # Flask-SocketIO client

# Connect to the server
sio = socketio.Client()
sio.connect("http://localhost:5000")

frame_lock = threading.Lock()
processed_frame = None  # Shared variable for storing received frames

@sio.on('matched_frame')
def on_processed_frame(data):
    """Receives processed frame from server"""
    global processed_frame
    try:
        np_arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Store frame for display
        with frame_lock:
            processed_frame = frame
    except Exception as e:
        print("Error displaying processed frame:", e)

def send_video():
    """Captures video and sends frames using emit"""
    cap = cv2.VideoCapture(0)  # Use webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Encode frame to Base64
        _, buffer = cv2.imencode('.jpg', frame)

        # Emit frame to server
        sio.emit('video_stream', buffer.tobytes())

    cap.release()

def display_video():
    """Continuously displays processed frames received from the server"""
    global processed_frame
    while True:
        with frame_lock:
            if processed_frame is not None:
                cv2.imshow("Processed Stream", processed_frame)
                cv2.waitKey(1)

if __name__ == "__main__":
    print("Connected to server!")

    # Start video sending in a separate thread
    # input("Press Enter to start sending video...\n")
    send_thread = threading.Thread(target=send_video)
    send_thread.start()

    # Start displaying processed frames
    display_video()
