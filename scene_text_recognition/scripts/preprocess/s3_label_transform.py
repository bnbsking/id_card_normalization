import json
import os
import random
import shutil

import opencc


random.seed(42)
cc = opencc.OpenCC('t2s.json')


def convert(ant_path: str, save_folder: str):
    with open(ant_path, "r", encoding="utf-8") as f:
        ant = json.load(f)
    random.shuffle(ant["data"])
    cut_idx = int(0.8 * len(ant["data"]))
    
    image_folder_train = os.path.join(save_folder, "train", "images")
    image_folder_valid = os.path.join(save_folder, "valid", "images")
    os.makedirs(image_folder_train, exist_ok=True)
    os.makedirs(image_folder_valid, exist_ok=True)
    with open(os.path.join(save_folder, "train", "train.txt"), "w", encoding="utf-8") as f:
        for data_dict in ant["data"][:cut_idx]:
            img_save_path = os.path.join(image_folder_train, os.path.basename(data_dict['data_path']))
            line = f"{img_save_path}\t{cc.convert(data_dict['gt_text'])}\n"
            f.write(line)
            shutil.copy(data_dict['data_path'], image_folder_train)
    with open(os.path.join(save_folder, "valid", "train.txt"), "w", encoding="utf-8") as f:
        for data_dict in ant["data"][cut_idx:]:
            img_save_path = os.path.join(image_folder_valid, os.path.basename(data_dict['data_path']))
            line = f"{img_save_path}\t{cc.convert(data_dict['gt_text'])}\n"
            f.write(line)
            shutil.copy(data_dict['data_path'], image_folder_valid)


if __name__ == "__main__":
    exp_name = "exp0"

    convert(
        f"/dataset/labels/text_recognition/{exp_name}/labels.json",
        f"/dataset/images/text_recognition/{exp_name}"
    )
