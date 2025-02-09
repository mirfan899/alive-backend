import collections
import os

import cv2
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# Connect to Milvus
from pymilvus import MilvusClient


client = MilvusClient("milvus_demo.db")
# Define Milvus collection schema
collection_name = "image_collection"
# Create or load collection
fields = [
    FieldSchema(name="image_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="descriptors", dtype=DataType.FLOAT_VECTOR, dim=32000)
]
schema = CollectionSchema(fields, description="Image descriptors")
# Create collection with the schema
if collection_name not in client.list_collections():
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="descriptors",
        index_type="IVF_FLAT",
        metric_type="L2",
        params={"nlist": 128}
    )
    client.create_collection(collection_name=collection_name, schema=schema)
    client.create_index(index_params=index_params, collection_name=collection_name)

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=1000,
        scaleFactor=1.2,
        nlevels=10,
        edgeThreshold=15,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31)

# Function to insert image descriptors into Milvus
def insert_image_to_milvus(image_path, video_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    if descriptors is not None:
        descriptors_flat = descriptors.astype(np.float32).flatten()

        # Pad to 32,000 dimensions if necessary
        if descriptors_flat.shape[0] < 32000:
            padding = np.zeros(32000 - descriptors_flat.shape[0], dtype=np.float32)
            descriptors_flat = np.concatenate([descriptors_flat, padding])

        client.insert(collection_name=collection_name, data=[{"descriptors": descriptors_flat.tolist(), "image_path": image_path}])
        image_video_mapping[image_path] = video_path


# Load images and videos from directories
image_dir = "images"
video_dir = "videos"
image_video_mapping = {}
for image_file in os.listdir(image_dir):
    if image_file.endswith(".jpg") or image_file.endswith(".png"):
        video_file = os.path.splitext(image_file)[0] + ".mp4"
        video_path = os.path.join(video_dir, video_file)
        if os.path.exists(video_path):
            insert_image_to_milvus(os.path.join(image_dir, image_file), video_path)
