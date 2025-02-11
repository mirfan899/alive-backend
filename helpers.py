import logging
import os

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from annoy import AnnoyIndex
from pymilvus import MilvusClient

# Paths to input images and their associated videos
# Connect to Milvus
client = MilvusClient("milvus_demo.db")


# Load collection
collection_name = "image_collection"

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=1000,
        scaleFactor=1.2,
        nlevels=10,
        edgeThreshold=15,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31)

IMAGE_VIDEO_MAPPING = {
    'images/image_1.jpg': './videos/image_1.mp4',
    'images/image_2.jpg': './videos/image_2.mp4'
}

# ------------------------ Feature Extraction ------------------------

def load_pretrained_model():
    """
    Load a pre-trained ResNet50 model and modify it to output feature vectors with adaptive pooling.
    """
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Replace the final average pooling with adaptive pooling
        model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # Remove the final classification layer
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        logging.info("Pre-trained ResNet50 model loaded successfully with adaptive pooling.")
        return model
    except Exception as e:
        logging.error(f"Failed to load pre-trained model: {e}")
        return None

def extract_feature(model, image_input):
    """
    Extract a feature vector from an image using the provided model.
    """
    try:
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                logging.error(f"Unable to load image {image_input}.")
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        else:
            logging.error("Unsupported image_input type.")
            return None

        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ResNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
        input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            feature = model(input_tensor).squeeze().numpy()
        # Normalize the feature vector
        feature = feature / np.linalg.norm(feature)
        return feature
    except Exception as e:
        logging.error(f"Exception in extract_feature: {e}")
        return None

# ------------------------ Vector Database (Annoy) ------------------------

def create_annoy_index(features, feature_dim=2048, n_trees=100):
    """
    Create and build an Annoy index.
    """
    try:
        index = AnnoyIndex(feature_dim, 'angular')
        for i, feature in enumerate(features):
            index.add_item(i, feature)
        index.build(n_trees)
        logging.info("Annoy index created and built successfully.")
        return index
    except Exception as e:
        logging.error(f"Exception in create_annoy_index: {e}")
        return None

def load_annoy_index(image_paths, model, feature_dim=2048, n_trees=100):
    """
    Load images, extract features, and create an Annoy index.
    """
    try:
        features = []
        id_to_video = {}
        for idx, img_path in enumerate(image_paths):
            if not os.path.exists(img_path):
                logging.error(f"Image file {img_path} does not exist.")
                continue

            feature = extract_feature(model, img_path)
            if feature is not None:
                features.append(feature)
                id_to_video[len(features)-1] = IMAGE_VIDEO_MAPPING.get(img_path, None)
                logging.info(f"Loaded and extracted features for {img_path}")
            else:
                logging.warning(f"Failed to extract features for {img_path}")
        if not features:
            logging.error("No features extracted. Exiting.")
            return None, None
        index = create_annoy_index(features, feature_dim=feature_dim, n_trees=n_trees)
        # Adjust id_to_video to match the Annoy index item indices
        adjusted_id_to_video = {i: id_to_video[i] for i in range(len(features))}
        return index, adjusted_id_to_video
    except Exception as e:
        logging.error(f"Exception in load_annoy_index: {e}")
        return None, None

# ------------------------ Utility Functions ------------------------

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_LINEAR):
    """
    Resizes an image while maintaining aspect ratio.
    """
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# ------------------------ Geometry Calculation Functions ------------------------

def calculate_length(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def calculate_slope(pt1, pt2):
    delta_x = pt2[0] - pt1[0]
    delta_y = pt2[1] - pt1[1]
    if delta_x == 0:
        return np.inf
    return delta_y / delta_x

def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# ------------------------ Parallelogram Validation ------------------------

def is_almost_parallelogram(dst, thresholds):
    try:
        if dst.shape != (4, 1, 2):
            logging.warning(f"Invalid shape for quadrilateral points: {dst.shape}")
            return False

        pts = dst.reshape(4, 2)
        ordered_pts = order_points_clockwise(pts)
        pt1, pt2, pt3, pt4 = ordered_pts

        length1 = calculate_length(pt1, pt2)
        length2 = calculate_length(pt3, pt4)
        length3 = calculate_length(pt2, pt3)
        length4 = calculate_length(pt4, pt1)

        slope1 = calculate_slope(pt1, pt2)
        slope2 = calculate_slope(pt3, pt4)
        slope3 = calculate_slope(pt2, pt3)
        slope4 = calculate_slope(pt4, pt1)

        length_diff_ratio_1 = abs(length1 - length2) / max(length1, length2) if max(length1, length2) != 0 else 0
        length_diff_ratio_2 = abs(length3 - length4) / max(length3, length4) if max(length3, length4) != 0 else 0

        slope_diff_1 = abs(slope1 - slope2)
        slope_diff_2 = abs(slope3 - slope4)

        if np.isinf(slope1) and np.isinf(slope2):
            slope_diff_1 = 0
        if np.isinf(slope3) and np.isinf(slope4):
            slope_diff_2 = 0

        logging.debug(f"Lengths: [{length1:.2f}, {length2:.2f}], [{length3:.2f}, {length4:.2f}]")
        logging.debug(f"Length Diff Ratios: [{length_diff_ratio_1:.2f}, {length_diff_ratio_2:.2f}]")
        logging.debug(f"Slopes: [{slope1:.2f}, {slope2:.2f}], [{slope3:.2f}, {slope4:.2f}]")
        logging.debug(f"Slope Diff: [{slope_diff_1:.2f}, {slope_diff_2:.2f}]")

        length_check = (length_diff_ratio_1 < thresholds['length_diff_ratio']) and \
                       (length_diff_ratio_2 < thresholds['length_diff_ratio'])

        parallelism_check = (slope_diff_1 < thresholds['parallelism_ratio']) and \
                            (slope_diff_2 < thresholds['parallelism_ratio'])

        if length_check and parallelism_check:
            logging.debug("Quadrilateral is almost a parallelogram.")
            return True
        else:
            logging.debug("Quadrilateral does not meet parallelogram criteria.")
            return False

    except Exception as e:
        logging.error(f"Exception in is_almost_parallelogram: {e}")
        return False

def is_valid_homography(homography, min_det=0.01):
    """
    Validate the homography matrix based on determinant and other criteria.

    Args:
        homography: The homography matrix.
        min_det: Minimum acceptable determinant to prevent degenerate transformations.

    Returns:
        Boolean indicating whether the homography is valid.
    """
    det = np.linalg.det(homography[:2, :2])
    if det < min_det:
        logging.warning(f"Invalid homography determinant: {det}")
        return False
    return True

# ------------------------ Augmented Reality Overlay ------------------------

def overlay_video(frame, frame_video, homography):
    """
    Overlay a video frame onto the original frame using the provided homography.
    """
    try:
        h, w, _ = frame_video.shape
        pts_video = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        warped_video = cv2.perspectiveTransform(pts_video, homography)
        warped_video_int = np.int32(warped_video)

        # Create a mask from the warped video polygon
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, warped_video_int, 255)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Warp the video frame
        warped_frame = cv2.warpPerspective(frame_video, homography, (frame.shape[1], frame.shape[0]))
        logging.debug("Video frame warped successfully.")

        # Combine the warped video with the original frame
        frame = cv2.bitwise_and(frame, cv2.bitwise_not(mask))
        frame = cv2.add(frame, warped_frame)
        logging.debug("Video frame blended with camera frame successfully.")

        return frame
    except Exception as e:
        logging.error(f"Exception in overlay_video: {e}")
        return frame


def process_and_insert_image(image2video:dict):
    image = cv2.imread(image2video["image_path"], cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = orb.detectAndCompute(image, None)

    if descriptors is not None:
        descriptors_flat = descriptors.astype(np.float32).flatten()

        # Pad descriptors to match 32000 dimensions
        if descriptors_flat.shape[0] < 32000:
            padding = np.zeros(32000 - descriptors_flat.shape[0], dtype=np.float32)
            descriptors_flat = np.concatenate([descriptors_flat, padding])

        # Insert descriptors into Milvus
        client.insert(collection_name=collection_name, data={"descriptors": [descriptors_flat.tolist()], "image_path": {image2video["image_path"]}, "video_path": image2video["video_path"]})
        return True
    else:
        return False


def search_image_in_milvus(feature, top_k=1):

    if feature.any():
        # Perform search in Milvus
        search_params = {
            "metric_type": "L2",  # ORB uses L2 norm for similarity
            "params": {"nlist": 10}  # nprobe controls the search scope
        }

        results = client.search(
            collection_name=collection_name,
            data=[feature],  # Pass the descriptors
            anns_field="descriptors",
            search_params=search_params,
            limit=top_k,
            output_fields=["image_id", "image_path"]  # You can retrieve additional fields if needed
        )

        if results.__len__() == 0:
            return None, None, None
        else:
            return results[0][0]['entity']["image_id"], results[0][0]['entity']["image_path"], results[0][0]['distance']  # Return the top results for further use

    else:
        print(f"No descriptors found.")
        return None, None, None
