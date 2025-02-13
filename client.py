import threading
import queue
import cv2
import numpy as np
import socketio

# Connect to the server
sio = socketio.Client()
sio.connect("http://localhost:5000")

frame_lock = threading.Lock()
processed_frame = None  # Shared variable for storing received frames
frame_queue = queue.Queue()  # Queue to store frames before sending

@sio.on('matched_frame')
def on_processed_frame(data):
    """Receives processed frame from server"""
    global processed_frame
    try:
        np_arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Store frame for display
        with frame_lock:
            processed_frame = frame.copy()  # Copy reduces lock contention
    except Exception as e:
        print("Error displaying processed frame:", e)

def capture_video():
    """Captures video and pushes frames to queue"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set frame rate to avoid overload

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Put frame in queue
        if not frame_queue.full():
            frame_queue.put(frame)

    cap.release()

def send_video():
    """Encodes and sends frames from queue"""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Encode frame to JPEG with quality 80 (faster encoding)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

            # Emit frame to server
            sio.emit('video_stream', buffer.tobytes())

def display_video():
    """Continuously displays processed frames received from the server"""
    global processed_frame
    while True:
        with frame_lock:
            if processed_frame is not None:
                cv2.imshow("Processed Stream", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Connected to server!")

    # Start threads for video capture and sending
    capture_thread = threading.Thread(target=capture_video, daemon=True)
    send_thread = threading.Thread(target=send_video, daemon=True)

    capture_thread.start()
    send_thread.start()

    # Start displaying processed frames
    display_video()
