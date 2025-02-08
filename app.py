import logging
import os
import threading

import cv2
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS

from helpers import load_pretrained_model, IMAGE_VIDEO_MAPPING, load_annoy_index, extract_feature, is_valid_homography, \
    is_almost_parallelogram, resize_with_aspect_ratio, overlay_video

# ------------------------ Configuration ------------------------

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


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


# ------------------------ Main Application ------------------------

def main():
    # Load the pre-trained model
    logging.info("Loading pre-trained model...")
    model = load_pretrained_model()
    if model is None:
        logging.error("Model loading failed. Exiting.")
        return

    # Prepare image paths
    image_paths = list(IMAGE_VIDEO_MAPPING.keys())

    # Load Annoy index and mapping
    logging.info("Creating Annoy index...")
    index, id_to_video = load_annoy_index(image_paths, model, feature_dim=FEATURE_DIM, n_trees=N_TREES)
    if index is None:
        logging.error("Failed to create Annoy index. Exiting.")
        return

    # Initialize ORB for keypoint detection and matching
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
    logging.info("Initialized ORB detector and BFMatcher.")

    # Precompute keypoints and descriptors for each image
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

    # Open the device camera
    logging.info("Accessing the camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Unable to access the camera.")
        return
    logging.info("Camera accessed successfully.")

    # Initialize video capture objects for each video
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

    # Set up display window
    cv2.namedWindow('AR Overlay', cv2.WINDOW_NORMAL)

    logging.info("Starting real-time AR overlay. Press 'q' to exit.")

    # Initialize match stability tracking
    current_match = {
        'idx': -1,
        'count': 0,
        'homography': None
    }

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to grab frame from camera.")
                break

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
                cv2.imshow('AR Overlay', frame_with_keypoints)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

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
                cv2.imshow('AR Overlay', frame_with_keypoints)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

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
                                            frame_video_resized = resize_with_aspect_ratio(frame_video, width=w_img, height=h_img)
                                            logging.debug(f"Resized video frame to: {frame_video_resized.shape}")

                                            # Overlay the video frame onto the detected image
                                            frame = overlay_video(frame, frame_video_resized, homography)

                                            # Optional: Display a message
                                            cv2.putText(frame, "Overlay Applied", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                                        1, (0, 255, 0), 2, cv2.LINE_AA)
                                            # else:
                                            #     logging.warning("Detected quadrilateral is not a parallelogram. Skipping overlay.")
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

            # Display the processed frame with keypoints
            cv2.imshow('AR Overlay', frame_with_keypoints)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Exit command received. Exiting.")
                break

    except KeyboardInterrupt:
        logging.info("Interrupted by user")

    except cv2.error as e:
        logging.error(f"OpenCV Error: {e}")

    finally:
        # Release all resources
        cap.release()
        for vc in video_caps.values():
            if vc is not None:
                vc.release()
        cv2.destroyAllWindows()
        logging.info("Released all resources and closed windows.")

# ------------------------ Flask API Setup ------------------------

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://alive-frontend-omega.vercel.app"}})

# Global variables to track initialization
index = None
id_to_video = None
initialization_success = False
initialization_error = None
overlay_lock = threading.Lock()

@app.route('/', methods=['GET','POST'])
def home():
    return jsonify({'message': 'Welcome to the AR Overlay API!'})

@app.route('/start-ar', methods=['POST','OPTIONS'])
def start_ar():
    global index, id_to_video, initialization_success, initialization_error
    try:
        logging.info("Received request to start AR Overlay.")
        with overlay_lock:
            if initialization_success:
                logging.info("AR Overlay is already initialized.")
                return jsonify({'status': 'AR Overlay Already Initialized'})
            elif initialization_error:
                logging.error(f"Initialization failed previously: {initialization_error}")
                return jsonify({'status': 'Initialization Failed', 'error': initialization_error}), 500

            # Start the AR overlay in a separate thread to prevent blocking
            thread = threading.Thread(target=main)
            thread.start()
            logging.info("AR Overlay initialization started in a new thread.")
            return jsonify({'status': 'AR Overlay Initialization Started'})
    except Exception as e:
        logging.error(f"Error in /start-ar endpoint: {e}")
        return jsonify({'status': 'Failed to start AR Overlay', 'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
