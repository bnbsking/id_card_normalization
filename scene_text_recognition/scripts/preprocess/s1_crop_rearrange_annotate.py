from collections import defaultdict
import copy
import json
import os
from typing import Dict, List, Tuple

import cv2
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
    def get_upper_leftest_box(self, boxes: List[List[int]]) -> List[int]:
        return [min(boxes, key=lambda box: (box[1], box[0]))] if boxes else []

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
        img_path: str,
        replace_folder: Tuple[str, str],
        str_ant_data: List | None = None
    ) -> List[str]:
    img = cv2.imread(img_path)
    for key, boxes in key2boxes.items():
        for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
            cropped_img = img[ymin:ymax, xmin:xmax, :]
            save_path = img_path.replace(replace_folder[0], replace_folder[1])
            prefix, ext = os.path.splitext(save_path)
            save_path = f"{prefix}_{key}_{i}{ext}"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, cropped_img)
            if str_ant_data is not None:
                str_ant_data.append({"data_path": save_path, "gt_text": ""})


if __name__ == "__main__":
    exp_name = "exp0"
    black = {"20121907189189.jpg"}
    url = "http://10.32.210.230:8061/upload"
    
    ant_path = f"/dataset/labels/pair_detection_transform/{exp_name}/.others/general.json"
    filter_dict = {"date": ["get_upper_leftest_box"]}
    replace_folder = ("/crop/", "/pair_crop/")
    save_str_ant_path = f"/dataset/labels/text_recognition/{exp_name}/labels.json"

    with open(ant_path, "r") as f:
        annotations = json.load(f)
        categories = annotations["categories"][1:]
        img_path_list = [ant["img_path"] for ant in annotations["data"]]
    str_ant = {"data": []} 
    for img_path in img_path_list:
        if os.path.basename(img_path) not in black:
            crop_info = request_crop_info(img_path, url)
            key2boxes = crop_info_by_class(crop_info, categories)
            key2boxes = CropFilter().run(key2boxes, filter_dict)
            crop_images(key2boxes, img_path, replace_folder, str_ant["data"])

    with open(save_str_ant_path, "w", encoding="utf-8") as f:
        json.dump(str_ant, f, ensure_ascii=False, indent=4)
    