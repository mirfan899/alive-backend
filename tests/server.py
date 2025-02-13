import time

from flask import Flask
from flask_socketio import SocketIO
import cv2
import base64
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route('/')
def index():
    return "WebSocket Video Processing Server Running"


@socketio.on('video_stream')  # Custom event instead of default "message"
def handle_video_frame(image_data):
    """Receives a frame, processes it, and sends it back using emit."""
    try:
        # Decode Base64 frame (No need to split anymore)
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        print("Received frame")
        # Processing: Convert to grayscale
        # processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Encode back to Base64
        _, buffer = cv2.imencode('.jpg', frame)
        # encoded_frame = base64.b64encode(buffer)
        time.sleep(0.1)
        # Send processed frame back using emit
        socketio.emit('matched_frame', buffer.tobytes())  # Send only Base64 data
        print("processed frame sent")
    except Exception as e:
        print("Error processing frame:", e)


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
