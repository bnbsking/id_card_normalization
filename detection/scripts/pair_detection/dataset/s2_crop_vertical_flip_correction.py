import json
import os
from typing import Tuple

import cv2
import numpy as np
import requests


def request_vertical_flip(url: str, img_path: str) -> bool:
    files = {"file": open(img_path, "rb")}
    data = {"meta": json.dumps({"model_key": "vertical_flip_classification"})}
    response = requests.post(url, files=files, data=data)
    assert response.status_code == 200
    result = response.json()
    return result["pd_cls"] == 1
    

def vertical_flip(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 0)  # Flip vertically


def main(url: str, ant_path: str, replace_folder: Tuple[str, str] = ("crop", "crop_vflip")):
    with open(ant_path, "r") as f:
        annotations = json.load(f)

    for ann in annotations["data"]:
        img_path = ann["data_path"]
        flip = request_vertical_flip(url, img_path)
        img = cv2.imread(img_path)
        if flip:
            img = cv2.flip(img, 0)
        new_img_path = img_path.replace(replace_folder[0], replace_folder[1])
        os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
        cv2.imwrite(new_img_path, img)


if __name__ == "__main__":
    url = "http://10.32.210.230:8062/upload"
    ant_path = "/dataset/labels/classification/exp2/labeled.json"
    main(url, ant_path)
