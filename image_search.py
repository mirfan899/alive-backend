from pymilvus import connections, Collection, MilvusClient

from helpers import load_pretrained_model, extract_feature

# Connect to Milvus
client = MilvusClient("milvus_demo.db")

# Load collection
collection_name = "image_collection"

model = load_pretrained_model()

def search_image_in_milvus(query_image_path, top_k=1):
    # Load and process the query image
    descriptors = extract_feature(model, query_image_path)
    if descriptors.any():
        # Perform search in Milvus
        search_params = {
            "metric_type": "L2",  # ORB uses L2 norm for similarity
            "params": {"nlist": 10}  # nprobe controls the search scope
        }

        results = client.search(
            collection_name=collection_name,
            data=[descriptors],  # Pass the descriptors
            anns_field="descriptors",
            search_params=search_params,
            limit=top_k,
            output_fields=["image_id", "image_path"]  # You can retrieve additional fields if needed
        )

        # Display results
        print(f"Top {top_k} matches for {query_image_path}:")
        for hit in results[0]:  # results[0] because we only searched one image
            print(f"Image Path: {hit['entity']['image_path']}, Distance: {hit['distance']}")

        if results.__len__() == 0:
            return None, None, None
        else:
            return results[0][0]['entity']["image_id"], results[0][0]['entity']["image_path"], results[0][0]['distance']  # Return the top results for further use
    else:
        print(f"No descriptors found in {query_image_path}.")
        return None, None, None


# Example Usage
# query_image_path = "images/image_1.jpg"
query_image_path = "test_images/image_1.png"
search_results = search_image_in_milvus(query_image_path)
