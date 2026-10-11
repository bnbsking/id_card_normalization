import copy
import json
import os
import random
from typing import Dict, List, Tuple

from cosmodules.detection.visualization.show import show_general
from tqdm import tqdm


random.seed(42)


def add_prefix_to_image_file_name(annotations: Dict, prefix: str, save_path: str):
    for img_dict in annotations["images"]:
        img_dict["file_name"] = os.path.join(prefix, img_dict["file_name"])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=4, ensure_ascii=False)


def visualize(ant_path: str, save_folder: str):
    with open(ant_path, "r") as f:
        general = json.load(f)
    img_path_list = [ant["img_path"] for ant in general["data"]]
    for img_path in tqdm(img_path_list):
        show_general(
            img_path=img_path,
            ant_path=ant_path,
            save_path=os.path.join(save_folder, os.path.basename(img_path))
        )


def get_annotations_and_indices(ant_path: str) -> Tuple[Dict, List[int], List[int]]:
    with open(ant_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    img_indices = list(range(len(annotations["data"])))
    random.shuffle(img_indices)
    split_idx = int(0.8 * len(img_indices))
    img_indices_train = img_indices[:split_idx]
    img_indices_valid = img_indices[split_idx:]
    return annotations, img_indices_train, img_indices_valid


def get_split_annotations(
        annotations: Dict,
        img_indices_train: List[int],
        img_indices_valid: List[int]
    ) -> Tuple[Dict, Dict]:
    annotations_train = copy.deepcopy(annotations)
    annotations_valid = copy.deepcopy(annotations)

    annotations_train["data"] = [data_dict for i, data_dict in enumerate(annotations["data"]) if i in img_indices_train]
    annotations_valid["data"] = [data_dict for i, data_dict in enumerate(annotations["data"]) if i in img_indices_valid]
    return annotations_train, annotations_valid