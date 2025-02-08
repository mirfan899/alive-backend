from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import uvicorn
import threading
import io

app = FastAPI()

# Load reference images and their corresponding videos
image_video_mapping = {
    "images/image_1.jpg": "videos/image_1.mp4",
    "images/image_2.jpg": "videos/image_2.mp4",
}

# Initialize ORB detector
orb = cv2.ORB_create()

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

# Initialize BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Global variables for video processing
current_video = None
video_capture = None
cap = None


def video_processing():
    global cap, video_capture, current_video

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints_frame, descriptors_frame = orb.detectAndCompute(gray_frame, None)

        if descriptors_frame is None:
            continue

        for ref_image_path, data in reference_data.items():
            if data["descriptors"] is None:
                continue

            if descriptors_frame.dtype != data["descriptors"].dtype:
                descriptors_frame = descriptors_frame.astype(data["descriptors"].dtype)

            matches = bf.match(data["descriptors"], descriptors_frame)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) > 20:
                src_pts = np.float32([data["keypoints"][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                h, w = data["image"].shape
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                frame = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

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

        if video_capture and video_capture.isOpened():
            ret_vid, video_frame = video_capture.read()
            if ret_vid and 'dst' in locals():
                video_frame_resized = cv2.resize(video_frame, (w, h))
                warped_video = cv2.warpPerspective(video_frame_resized, M, (frame.shape[1], frame.shape[0]))

                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.int32(dst), 255)
                mask_inv = cv2.bitwise_not(mask)

                frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                video_fg = cv2.bitwise_and(warped_video, warped_video, mask=mask)

                final_frame = cv2.add(frame_bg, video_fg)
            else:
                final_frame = frame
        else:
            final_frame = frame

        _, jpeg = cv2.imencode('.jpg', final_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()
    if video_capture:
        video_capture.release()
    cv2.destroyAllWindows()


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(video_processing(), media_type='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
