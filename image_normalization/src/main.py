import json
from tempfile import TemporaryDirectory
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image, ImageOps
import requests

from src.crop_card_to_json import main as crop_card_to_json_main


class Normalization:
    def __init__(self, detection_url: str, classification_url: str, categories: List[str], str_url: str):
        self.detection_url = detection_url
        self.classification_url = classification_url
        self.categories = categories
        self.str_url = str_url

    @staticmethod
    def remove_exif(img_path: str) -> np.ndarray:
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)
        return np.array(img)

    @staticmethod
    def detection_crop(url: str, img_path: str) -> np.ndarray:
        files = {"file": open(img_path, "rb")}
        data = {"meta": json.dumps({"model_key": "detection_model"})}
        response = requests.post(url, files=files, data=data)
        out = response.json()

        img = cv2.imread(img_path)
        if out:  # crop
            img = img[out['ymin']:out['ymax'], out['xmin']:out['xmax']]  
        return img

    @staticmethod
    def rotate2horizon(img: np.ndarray) -> np.ndarray:
        height, width = img.shape[:2]
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) if height > width else img

    @staticmethod
    def vflip(url: str, img_path: str) -> np.ndarray:
        files = {"file": open(img_path, "rb")}
        data = {"meta": json.dumps({"model_key": "vertical_flip_classification"})}
        response = requests.post(url, files=files, data=data)
        out = response.json()
        
        img = cv2.imread(img_path)
        if out and out.get("pd_cls", 0) == 1:  # vflip
            img = cv2.flip(img, 0)
        return img

    def run(self, img_path: str) -> np.ndarray:
        with TemporaryDirectory() as tmp_dir:
            img = self.remove_exif(img_path)
            save_path = f"{tmp_dir}/remove_exif.jpg"
            cv2.imwrite(save_path, img)

            img = self.detection_crop(self.detection_url, save_path)
            img = self.rotate2horizon(img)
            save_path = f"{tmp_dir}/detection_crop.jpg"
            cv2.imwrite(save_path, img)

            img = self.vflip(self.classification_url, save_path)
            return img

    def run_full(self, img_path: str) -> Dict[str, str]:
        with TemporaryDirectory() as tmp_dir:
            img = self.remove_exif(img_path)
            save_path = f"{tmp_dir}/remove_exif.jpg"
            cv2.imwrite(save_path, img)

            img = self.detection_crop(self.detection_url, save_path)
            img = self.rotate2horizon(img)
            save_path = f"{tmp_dir}/detection_crop.jpg"
            cv2.imwrite(save_path, img)

            img = self.vflip(self.classification_url, save_path)
            save_path = f"{tmp_dir}/vflip.jpg"  # from now on add more
            cv2.imwrite(save_path, img)

            result_dict = crop_card_to_json_main(
                self.detection_url,
                save_path,
                self.categories,
                self.str_url
            )
            return result_dict
    