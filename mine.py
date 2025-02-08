import cv2
import numpy as np

# Load reference images and their corresponding videos
image_video_mapping = {
    "images/image_1.jpg": "videos/image_1.mp4",
    "images/image_2.jpg": "videos/image_2.mp4",
}

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=1000)

# Load reference images and compute keypoints/descriptors
reference_data = {}
for image_path, video_path in image_video_mapping.items():
    image = cv2.imread(image_path, 0)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    reference_data[image_path] = {
        "image": image,
        "keypoints": keypoints,
        "descriptors": descriptors,
        "video": video_path
    }

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

current_video = None
video_capture = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors in the current frame
    keypoints_frame, descriptors_frame = orb.detectAndCompute(gray_frame, None)

    if descriptors_frame is None:
        cv2.imshow('Augmented Reality', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    for ref_image_path, data in reference_data.items():
        if data["descriptors"] is None:
            continue

        # Ensure descriptors have the same data type
        if descriptors_frame.dtype != data["descriptors"].dtype:
            descriptors_frame = descriptors_frame.astype(data["descriptors"].dtype)

        matches = bf.match(data["descriptors"], descriptors_frame)
        matches = sorted(matches, key=lambda x: x.distance)

        # Check if matches are good enough
        if len(matches) > 20:  # Adjust threshold as needed
            frame = cv2.putText(frame, f"Match Found: {ref_image_path}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Find homography to get the position of the detected image
            src_pts = np.float32([data["keypoints"][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = data["image"].shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # Draw the detected area
            frame = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

            # Play video if not already playing
            if current_video != ref_image_path:
                if video_capture:
                    video_capture.release()

                video_capture = cv2.VideoCapture(data["video"])
                current_video = ref_image_path
            break
    else:
        current_video = None
        if video_capture:
            video_capture.release()
            video_capture = None

    # Overlay the video on the detected area if playing
    if video_capture and video_capture.isOpened():
        ret_vid, video_frame = video_capture.read()
        if ret_vid and 'dst' in locals():
            # Warp the video frame to the detected image area
            video_frame_resized = cv2.resize(video_frame, (w, h))
            warped_video = cv2.warpPerspective(video_frame_resized, M, (frame.shape[1], frame.shape[0]))

            # Create mask for the warped video
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(dst), 255)
            mask_inv = cv2.bitwise_not(mask)

            # Black-out the area on the frame where the video will be placed
            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            video_fg = cv2.bitwise_and(warped_video, warped_video, mask=mask)

            # Combine the frame and the video
            final_frame = cv2.add(frame_bg, video_fg)
            cv2.imshow('Augmented Reality', final_frame)
        else:
            video_capture.release()
            video_capture = None
            cv2.imshow('Augmented Reality', frame)
    else:
        cv2.imshow('Augmented Reality', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
if video_capture:
    video_capture.release()
cv2.destroyAllWindows()