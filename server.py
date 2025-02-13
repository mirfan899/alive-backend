import base64
import logging
import os

from flask import Flask, Response
from flask_socketio import SocketIO
import numpy as np
import cv2

from helpers import load_pretrained_model, IMAGE_VIDEO_MAPPING, load_annoy_index, extract_feature, \
    resize_with_aspect_ratio, overlay_video, is_valid_homography

# import faiss
# Parameters
FEATURE_DIM = 2048
N_TREES = 50  # Increased for better accuracy
DISTANCE_THRESHOLD = 1.5
VIDEO_PLAYBACK_SPEED = 1.0

# Parallelogram Thresholds
parallelogram_thresholds = {
    'length_diff_ratio': 1.0,
    'parallelism_ratio': 1.0
}

# Stability Parameters
MATCH_STABILITY_THRESHOLD = 3  # Number of consecutive frames required for a stable match
GOOD_MATCH_DISTANCE_THRESHOLD = 35  # Adjust as needed
model = load_pretrained_model()
image_paths = list(IMAGE_VIDEO_MAPPING.keys())
orb = cv2.ORB_create(
        nfeatures=1000,
        scaleFactor=1.2,
        nlevels=10,
        edgeThreshold=15,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31
    )
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


# Load Annoy index and mapping
logging.info("Creating Annoy index...")
index, id_to_video = load_annoy_index(image_paths, model, feature_dim=FEATURE_DIM, n_trees=N_TREES)
if index is None:
    logging.error("Failed to create Annoy index. Exiting.")

image_kp_desc = {}
for idx, img_path in enumerate(image_paths):
    if not os.path.exists(img_path):
        logging.error(f"Image file {img_path} does not exist for keypoints.")
        continue
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logging.error(f"Unable to load image {img_path} for keypoints.")
        continue
    kp, desc = orb.detectAndCompute(image, None)
    if desc is not None:
        image_kp_desc[idx] = (kp, desc)
        logging.info(f"Computed keypoints and descriptors for {img_path}")
    else:
        logging.warning(f"No descriptors found for {img_path}")

video_caps = {}
for idx, video_path in id_to_video.items():
    if not os.path.exists(video_path):
        logging.error(f"Video file {video_path} does not exist.")
        video_caps[idx] = None
        continue
    cap_video = cv2.VideoCapture(video_path)
    if not cap_video.isOpened():
        logging.error(f"Unable to open video {video_path}.")
        video_caps[idx] = None
    else:
        # Adjust video playback speed if necessary
        if VIDEO_PLAYBACK_SPEED != 1.0:
            fps = cap_video.get(cv2.CAP_PROP_FPS)
            cap_video.set(cv2.CAP_PROP_FPS, fps * VIDEO_PLAYBACK_SPEED)
        video_caps[idx] = cap_video
        logging.info(f"Opened video {video_path} for overlay.")

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # Enable CORS for all origins


@socketio.on('video_stream')
def handle_video_stream(data):
    """Handles incoming video frames from the client."""
    frame_data = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
    current_match = {
        'idx': -1,
        'count': 0,
        'homography': None
    }
    # Extract global feature from the current frame
    feature = extract_feature(model, frame)
    if feature is None:
        logging.warning("Failed to extract features from the current frame.")
        # Still compute keypoints for visualization
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, desc_frame = orb.detectAndCompute(gray_frame, None)
        # Draw keypoints as center points (dots)
        frame_with_keypoints = frame.copy()
        if kp_frame is not None:
            for kp in kp_frame:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(frame_with_keypoints, (x, y), 2, (0, 255, 0), -1)  # Small dot
        # cv2.imshow('AR Overlay', frame_with_keypoints)
        _, buffer = cv2.imencode('.jpg', frame_with_keypoints)
        socketio.emit("matched_frame", buffer.tobytes())
    # Search in Annoy index for the nearest neighbor
    nearest_indices, distances = index.get_nns_by_vector(feature, 1, include_distances=True)
    if not nearest_indices:
        logging.debug("No nearest neighbors found.")
        # Still compute keypoints for visualization
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, desc_frame = orb.detectAndCompute(gray_frame, None)
        # Draw keypoints as center points (dots)
        frame_with_keypoints = frame.copy()
        if kp_frame is not None:
            for kp in kp_frame:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(frame_with_keypoints, (x, y), 2, (0, 255, 0), -1)  # Small dot
        _, buffer = cv2.imencode('.jpg', frame_with_keypoints)
        socketio.emit("matched_frame", buffer.tobytes())

    match_idx = nearest_indices[0]
    distance = distances[0]
    logging.debug(f"Match Index: {match_idx}, Distance: {distance}")

    # Define a distance threshold for matching
    if distance < DISTANCE_THRESHOLD and match_idx in id_to_video and id_to_video[match_idx]:
        if match_idx == current_match['idx']:
            current_match['count'] += 1
        else:
            current_match['idx'] = match_idx
            current_match['count'] = 1
        logging.info(f"{current_match['count']}.....")
        if current_match['count'] >= MATCH_STABILITY_THRESHOLD:
            best_match_idx = current_match['idx']
            video_path = id_to_video[best_match_idx]
            video_cap = video_caps.get(best_match_idx, None)
            if video_cap is not None:
                # Read next frame from the video
                ret_video, frame_video = video_cap.read()
                if not ret_video:
                    logging.info(f"Reached end of video {video_path}. Restarting.")
                    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_video, frame_video = video_cap.read()
                if ret_video:
                    logging.debug(f"Video frame read successfully from {video_path}.")

                    # Detect keypoints and compute descriptors in the camera frame
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    kp_frame, desc_frame = orb.detectAndCompute(gray_frame, None)
                    if desc_frame is not None and best_match_idx in image_kp_desc:
                        # Get keypoints and descriptors of the matched image
                        kp_image, desc_image = image_kp_desc.get(best_match_idx, (None, None))
                        if desc_image is not None:
                            # Match descriptors
                            matches = bf.match(desc_image, desc_frame)
                            matches = sorted(matches, key=lambda x: x.distance)
                            logging.debug(f"Number of Matches: {len(matches)}")

                            # Apply a good match distance threshold
                            good_matches = [m for m in matches if m.distance < GOOD_MATCH_DISTANCE_THRESHOLD]
                            logging.debug(f"Number of Good Matches: {len(good_matches)}")

                            if len(good_matches) > GOOD_MATCH_DISTANCE_THRESHOLD:  # Increased threshold for better homography
                                # Compute homography
                                src_pts = np.float32([kp_image[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                                if homography is not None and is_valid_homography(homography):
                                    logging.info("Homography computed successfully.")

                                    # Define the corners of the image
                                    h_img, w_img = cv2.imread(image_paths[best_match_idx], cv2.IMREAD_GRAYSCALE).shape
                                    pts = np.float32([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]]).reshape(-1, 1, 2)
                                    dst = cv2.perspectiveTransform(pts, homography)

                                    # Validate if the detected quadrilateral is almost a parallelogram
                                    # if is_almost_parallelogram(dst, parallelogram_thresholds):
                                    # Draw the detected region on the frame in green
                                    if dst is not None and len(dst) == 4:
                                        cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

                                    # Resize the video frame to match the detected image's size
                                    frame_video_resized = resize_with_aspect_ratio(frame_video, width=w_img,
                                                                                   height=h_img)
                                    logging.debug(f"Resized video frame to: {frame_video_resized.shape}")

                                    # Overlay the video frame onto the detected image
                                    frame = overlay_video(frame, frame_video_resized, homography)
                                    _, buffer = cv2.imencode('.jpg', frame)
                                    socketio.emit("matched_frame", buffer.tobytes())
                                else:
                                    logging.warning("Homography could not be computed or is invalid.")
                            else:
                                logging.warning("Not enough good matches to compute homography.")
                else:
                    logging.warning(f"Failed to read frame from video {video_path}.")
    else:
        # Reset match counter if no valid match
        current_match['idx'] = -1
        current_match['count'] = 0

    # Detect keypoints on every frame for visualization
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, desc_frame = orb.detectAndCompute(gray_frame, None)

    # Draw keypoints as center points (dots) on the camera frame for visualization
    frame_with_keypoints = frame.copy()
    if kp_frame is not None:
        for kp in kp_frame:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(frame_with_keypoints, (x, y), 2, (0, 255, 0), -1)  # Small dot

    _, buffer = cv2.imencode('.jpg', frame_with_keypoints)
    socketio.emit("matched_frame", buffer.tobytes())

@app.route('/')
def homepage():
    return "Server is running and ready to receive video stream."


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
