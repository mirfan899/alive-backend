from pymilvus import MilvusClient, DataType

from tests.helpers import load_pretrained_model, extract_feature

CLUSTER_ENDPOINT = "http://192.154.253.160:19530"
TOKEN = "root:Milvus"

client = MilvusClient(
    uri=CLUSTER_ENDPOINT,
    token=TOKEN
)

# Load collection
collection_name = "image_search"

model = load_pretrained_model()

def search_image_in_milvus(query_image_path, top_k=5):
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
            output_fields=["id", "image_path", "video_path",]  # You can retrieve additional fields if needed
        )

        # Display results
        print(f"Top {top_k} matches for {query_image_path}:")
        for hit in results[0]:  # results[0] because we only searched one image
            print(f" id {hit['entity']['id']}Image Path: {hit['entity']['image_path']}, Distance: {hit['distance']}")

        if results.__len__() == 0:
            return None, None, None
        else:
            return results[0][0]['entity']["image_path"], results[0][0]['entity']["video_path"], results[0][0]['distance']  # Return the top results for further use
    else:
        print(f"No descriptors found in {query_image_path}.")
        return None, None, None


# Example Usage
# query_image_path = "images/image_1.jpg"
query_image_path = "../test_images/image_1.png"
search_results = search_image_in_milvus(query_image_path)
