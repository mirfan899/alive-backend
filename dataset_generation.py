import logging
import os

import cv2

from helpers import extract_feature, IMAGE_VIDEO_MAPPING, create_annoy_index, load_pretrained_model

model = load_pretrained_model()
FEATURE_DIM = 2048
N_TREES = 100  # Increased for better accuracy
DISTANCE_THRESHOLD = 1.0
VIDEO_PLAYBACK_SPEED = 1.0

# Parallelogram Thresholds
parallelogram_thresholds = {
    'length_diff_ratio': 0.75,
    'parallelism_ratio': 0.75
}

import chromadb
# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
client = chromadb.Client()

collection = client.create_collection("image2video")

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
                id_to_video[len(features) - 1] = IMAGE_VIDEO_MAPPING.get(img_path, None)
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

def main():
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
        nfeatures=15000,
        scaleFactor=1.2,
        nlevels=20,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31
    )
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
    collection.add(
        documents=["This is document1", "This is document2"],
        # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
        metadatas=[{"source": "notion"}, {"source": "google-docs"}],  # filter on these!
        ids=["doc1", "doc2"],  # unique for each doc
    )
    print(image_kp_desc)

if __name__ == "__main__":
    main()