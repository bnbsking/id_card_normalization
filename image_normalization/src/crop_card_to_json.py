from collections import defaultdict
import copy
import json
from tempfile import TemporaryDirectory
from typing import Dict, List

import cv2
import numpy as np
import requests


def request_crop_info(img_path: str, url: str) -> Dict[str, List]:
    """
    Return example
        {
            'boxes': [
                [910, 1421, 1521, 1527],
                [905, 777, 1394, 868],
                [895, 999, 1393, 1084],
                [1902, 785, 1986, 884],
                [1476, 999, 1979, 1089],
                [900, 420, 1280, 530]
            ],
            'confidence': [0.968, 0.94, 0.94, 0.939, 0.936, 0.931],
            'class_id': [3, 2, 2, 1, 2, 0]
        }
    """
    model_key = "pair_detection_model"
    files = {"file": open(img_path, "rb")}
    data = {"meta": json.dumps({"model_key": model_key})}
    response = requests.post(url, files=files, data=data)
    crop_info = response.json()
    return crop_info


def crop_info_by_class(crop_info: Dict[str, List], categories: List[str]) -> Dict[str, List[List[int]]]:
    """
    Return example:
    {
        'address': [[629, 803, 1837, 1147]],
        'id_num': [[1111, 1312, 2714, 1550]], 
        'date': [[645, 611, 1557, 765]], 
        'name': [[639, 241, 1027, 376]], 
        'gender': [[639, 458, 749, 575]]
    }
    """
    key2boxes = defaultdict(list)
    for box, cid in zip(crop_info["boxes"], crop_info["class_id"]):
        key2boxes[categories[cid]].append(box)
    return key2boxes


class CropFilter:
    def leave_upper_leftest_box(self, boxes: List[List[int]]) -> List[int]:
        return [min(boxes, key=lambda box: (box[1], box[0]))] if boxes else []

    def sort_ymin_xmin(self, boxes: List[List[int]]) -> List[List[int]]:
        return sorted(boxes, key=lambda box: (box[1], box[0])) if boxes else []

    def run(
            self,
            key2boxes: Dict[str, List[List[int]]],
            filter_dict: Dict[str, List[str]]
        ) -> Dict[str, List[int]]:
        result = copy.deepcopy(key2boxes)
        for key in filter_dict:
            for method_name in filter_dict[key]:
                method = getattr(self, method_name)
                result[key] = method(key2boxes[key])
        return result


def crop_images(
        key2boxes: Dict[str, List[List[int]]],
        img_path: str
    ) -> List[str]:
    img = cv2.imread(img_path)
    key2imgs = defaultdict(list)
    for key, boxes in key2boxes.items():
        for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
            cropped_img = img[ymin:ymax, xmin:xmax, :]
            key2imgs[key].append(cropped_img)
    return key2imgs


def request_str(img: np.ndarray, url: str) -> str:
    with TemporaryDirectory() as tmpdir:
        tmp_img_path = f"{tmpdir}/temp.jpg"
        cv2.imwrite(tmp_img_path, img)

        files = {"file": open(tmp_img_path, "rb")}
        data = {"meta": json.dumps({"model_key": "svtrv2"})}
        response = requests.post(url, files=files, data=data)
        str_text = response.json()
        return str_text


def get_result(key2imgs: Dict[str, List[np.ndarray]], str_url: str) -> Dict[str, str]:
    result = defaultdict(str)
    for key, img_list in key2imgs.items():
        for img in img_list:
            result[key] += request_str(img, str_url)
    return result


def main(detection_url: str, img_path: str, categories: List[str], str_url: str):
    crop_info = request_crop_info(img_path, detection_url)
    key2boxes = crop_info_by_class(crop_info, categories)
    key2boxes = CropFilter().run(
        key2boxes,
        {
            "date": ["leave_upper_leftest_box"],
            "address": ["sort_ymin_xmin"]
        }
    )
    key2imgs = crop_images(key2boxes, img_path)
    result = get_result(key2imgs, str_url)
    return result
