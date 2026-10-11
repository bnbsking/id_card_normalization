import json
import os

from cosmodules.detection.format_conversion.dataset_conversion import (
    ConvertDatasetFromGeneral2Any
)

from src.dataset.labeled_to_ready_train import get_annotations_and_indices, get_split_annotations


if __name__ == "__main__":
    exp_name = "exp0"

    folder = f"/dataset/labels/pair_detection_transform/{exp_name}"
    
    annotations, img_indices_train, img_indices_valid = get_annotations_and_indices(
        os.path.join(folder, ".others", "general.json")
    )
    annotations_train, annotations_valid = get_split_annotations(annotations, img_indices_train, img_indices_valid)

    with open(os.path.join(folder, ".others", "general_train.json"), "w", encoding="utf-8") as f:
        json.dump(annotations_train, f, indent=4, ensure_ascii=False)
    with open(os.path.join(folder, ".others", "general_valid.json"), "w", encoding="utf-8") as f:
        json.dump(annotations_valid, f, indent=4, ensure_ascii=False)
    ConvertDatasetFromGeneral2Any.general2coco(
        ant_path=os.path.join(folder, ".others", "general_train.json"),
        save_folder=os.path.join(folder, "train"),
        img_path_use_basename_only=False,
        copy_images=False,
        save_name="_annotations.coco.json"
    )
    ConvertDatasetFromGeneral2Any.general2coco(
        ant_path=os.path.join(folder, ".others", "general_valid.json"),
        save_folder=os.path.join(folder, "valid"),
        img_path_use_basename_only=False,
        copy_images=False,
        save_name="_annotations.coco.json"
    )
