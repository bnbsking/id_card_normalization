import json
import os

from cosmodules.detection.metrics.export_worst import export_worst_cases
from cosmodules.detection.visualization.show import show_general


def vis(pred_path: str, save_folder: str):
    with open(pred_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    for data_dict in predictions["data"]:
        show_general(
            data_dict["img_path"],
            pred_path,
            os.path.join(save_folder, os.path.basename(data_dict["img_path"]))
        )


def run_export_worst_cases(pred_path: str, metrics_path: str, save_folder: str):
    with open(pred_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    data_path_list = [data_dict["img_path"] for data_dict in annotations["data"]]

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    sorted_accuracy_index_list = metrics["sorted_accuracy_index_list"]
    
    export_worst_cases(data_path_list, pred_path, sorted_accuracy_index_list, save_folder, top_n=10)
    