from pymilvus import connections, Collection, MilvusClient
import cv2
import numpy as np

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


def search_image_in_milvus(query_image_path, top_k=1):
    # Load and process the query image
    query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = orb.detectAndCompute(query_image, None)

    if descriptors is not None:
        descriptors_flat = descriptors.astype(np.float32).flatten()

        # Pad to 32,000 dimensions if necessary (since this is how we inserted)
        if descriptors_flat.shape[0] < 32000:
            padding = np.zeros(32000 - descriptors_flat.shape[0], dtype=np.float32)
            descriptors_flat = np.concatenate([descriptors_flat, padding])

        # Perform search in Milvus
        search_params = {
            "metric_type": "L2",  # ORB uses L2 norm for similarity
            "params": {"nlist": 10}  # nprobe controls the search scope
        }

        results = client.search(
            collection_name=collection_name,
            data=[descriptors_flat.tolist()],  # Pass the descriptors
            anns_field="descriptors",
            search_params=search_params,
            limit=top_k,
            output_fields=["image_id", "image_path"]  # You can retrieve additional fields if needed
        )

        # Display results
        print(f"Top {top_k} matches for {query_image_path}:")
        for hit in results[0]:  # results[0] because we only searched one image
            print(f"Image Path: {hit['entity']['image_path']}, Distance: {hit['distance']}")

        return results[0]  # Return the top results for further use
    else:
        print(f"No descriptors found in {query_image_path}.")
        return None


# Example Usage
query_image_path = "images/image_1.jpg"
search_results = search_image_in_milvus(query_image_path)
