from pymilvus import connections, Collection
import cv2
import numpy as np

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Load collection
collection_name = "image_collection"
collection = Collection(name=collection_name)

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=1000,
        scaleFactor=1.2,
        nlevels=10,
        edgeThreshold=15,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31)

# Function to search image in Milvus
def search_image_in_milvus(image_path):
    image = cv2.imread(image_path, 0)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    if descriptors is None:
        print("No descriptors found in image.")
        return

    descriptors_flat = descriptors.astype(np.float32).flatten()
    search_params = {
        "data": [descriptors_flat],
        "anns_field": "descriptors",
        "param": {"metric_type": "L2", "params": {"nprobe": 10}},
        "limit": 1
    }
    results = collection.search(**search_params)

    if results and results[0].distances[0] < 50.0:  # Adjust distance threshold as needed
        print(f"Match found with ID: {results[0].ids[0]}, Distance: {results[0].distances[0]}")
    else:
        print("No matching image found.")

# Example usage
search_image_in_milvus("query_image.jpg")
