from dataclasses import dataclass
import json
import os
from typing import Tuple

from PIL import Image
import requests


@dataclass
class Coordinates:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    confidence: float


def get_coordinates(img_path: str, url: str) -> Coordinates:
    files = {"file": open(img_path, "rb")}
    data = {"meta": json.dumps({"model_key": "detection_model"})}
    response = requests.post(url, files=files, data=data)
    assert response.status_code == 200
    result = response.json()  # e.g. {'xmin': 97, 'ymin': 1378, 'xmax': 2812, 'ymax': 3131, 'confidence': 0.777}
    try:
        return Coordinates(**result)
    except:
        img = Image.open(img_path)
        return Coordinates(xmin=0, ymin=0, xmax=img.width, ymax=img.height, confidence=0.0)


def crop_and_save(img_path: str, coordinates: Coordinates, replace_folder: Tuple[str, str]):
    img = Image.open(img_path)
    cropped_img = img.crop((coordinates.xmin, coordinates.ymin, coordinates.xmax, coordinates.ymax))
    
    if cropped_img.height > cropped_img.width:
        cropped_img = cropped_img.rotate(90, expand=True)
    
    save_path = img_path.replace(replace_folder[0], replace_folder[1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cropped_img.save(save_path)


def main(
        detection_ant_path: str,
        detection_crop_url: str,
        classification_save_ant_folder: str,
        replace_folder: Tuple[str, str] = ("remove_exif", "crop")
    ):
    with open(detection_ant_path, "r", encoding="utf-8") as f:
        detection_general = json.load(f)
    classification_general = {"categories": ["normal", "vflip"], "data": []}

    for data_dict in detection_general["data"]:
        img_path = data_dict["img_path"]
        coordinates = get_coordinates(img_path, detection_crop_url)
        crop_and_save(img_path, coordinates, replace_folder)
        classification_general["data"].append(
            {
                "data_path": img_path.replace(replace_folder[0], replace_folder[1]),
                "gt_cls": 0
            }
        )

    os.makedirs(classification_save_ant_folder, exist_ok=True)
    with open(os.path.join(classification_save_ant_folder, "unlabeled.json"), "w", encoding="utf-8") as f:
        json.dump(classification_general, f, indent=4, ensure_ascii=False)
    with open(os.path.join(classification_save_ant_folder, "labeled.json"), "w", encoding="utf-8") as f:
        json.dump(classification_general, f, indent=4, ensure_ascii=False)
