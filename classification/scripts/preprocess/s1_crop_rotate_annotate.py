import os

import yaml

from src.preprocess.crop_rotate_annotate import main


if __name__ == "__main__":
    exp_name = "exp2"

    with open(os.environ["BASE_CFG_PATH"], "r", encoding="utf-8") as f:
        detection_crop_url = yaml.safe_load(f)["preprocess"]["detection_crop_url"]
    main(
        f"/dataset/labels/detection_transform/{exp_name}/.others/general.json",
        detection_crop_url,
        f"/dataset/labels/classification/{exp_name}"
    )
