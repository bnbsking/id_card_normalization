import copy
import json
import os
import random


random.seed(42)


def split(input_path: str, output_folder: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data_list = data["data"]
    random.shuffle(data_list)
    split_idx = int(len(data_list) * 0.8)

    train_data = copy.deepcopy(data)
    train_data["data"] = data_list[:split_idx]
    valid_data = copy.deepcopy(data)
    valid_data["data"] = data_list[split_idx:]

    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open(os.path.join(output_folder, "valid.json"), "w", encoding="utf-8") as f:
        json.dump(valid_data, f, ensure_ascii=False, indent=4)
        