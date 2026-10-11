import json

from cosmodules.detection.format_conversion.dataset_conversion import (
    ConvertDatasetFromAnyToGeneral
)

from src.dataset.labeled_to_ready_train import add_prefix_to_image_file_name


if __name__ == "__main__":
    exp_name = "exp0"

    image_file_name_prefix = "/dataset/images/crop_vflip/"
    input_path = f"/dataset/labels/pair_detection/{exp_name}/instances_default.json"
    output_coco_path = f"/dataset/labels/pair_detection_transform/{exp_name}/.others/coco.json"
    output_general_path = f"/dataset/labels/pair_detection_transform/{exp_name}/.others/general.json"
    
    with open(input_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    add_prefix_to_image_file_name(annotations, image_file_name_prefix, output_coco_path)
    ConvertDatasetFromAnyToGeneral.coco2general(
        img_folder="",
        ant_path=output_coco_path,
        save_path=output_general_path
    )
