import os
import urllib.request

def download_file(url, destination):
    if not os.path.exists(destination):
        print(f"Downloading {destination}...")
        urllib.request.urlretrieve(url, destination)
        print(f"Downloaded {destination}")
    else:
        print(f"{destination} already exists.")

# YOLOv4-tiny weights and config URLs
yolo_weights_url = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"
yolo_cfg_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
coco_names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# Download files
download_file(yolo_weights_url, "yolov4-tiny.weights")
download_file(yolo_cfg_url, "yolov4-tiny.cfg")
download_file(coco_names_url, "coco.names")
