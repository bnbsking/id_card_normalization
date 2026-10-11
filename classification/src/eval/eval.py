import copy
import json
import os
from typing import Dict, List

from cosmodules.classification.metrics.export_worst import export_worst_cases
from cosmodules.classification.metrics.metrics_pipeline import ClassificationMetricsPipeline
from cosmodules.classification.metrics.plot_all import plot_all
import numpy as np
import torch

from src.dataset.vertical_flip import VerticalFlipClassificationDataset
from src.models.probing import TimmProbingModel


def load_model(model_name: str, num_classes: int, pretrain_weights: str) -> torch.nn.Module:
    model = TimmProbingModel(model_name, num_classes)
    model.load_state_dict(torch.load(pretrain_weights))
    return model


def inference(
        ant_path: str,
        img_size: int,
        model: torch.nn.Module,
        save_path: str,
        do_flip: bool = False,
    ):
    with open(ant_path, "r") as f:
        annotations = json.load(f)
    out = {
        "categories": annotations["categories"],
        "data": []
    }

    data_list = copy.deepcopy(annotations["data"])
    dset = VerticalFlipClassificationDataset(
        data_list=data_list,
        img_size=img_size,
        do_flip=do_flip
    )
    for img_idx in range(len(data_list)):
        augments = 1 + int(bool(do_flip))
        for i in range(augments):
            dataset_idx = img_idx * augments + i
            x = dset[dataset_idx][0].unsqueeze(0)
            with torch.no_grad():
                pred = model(x)
            pd_probs = torch.softmax(pred, dim=1)
            pd_probs = pd_probs.detach().cpu().numpy().tolist()[0]

            data_dict = copy.deepcopy(annotations["data"][img_idx])
            data_dict["gt_cls"] = (data_dict["gt_cls"] + i) % 2
            data_dict["pd_probs"] = pd_probs
            out["data"].append(data_dict)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)


def full_metrics_pipeline(pred_path: str, save_path: str) -> Dict:
    with open(pred_path, "r", encoding="utf-8") as f:
        infer_out = json.load(f)

    labels = np.array(
        [data_dict["gt_cls"] for data_dict in infer_out["data"]]
    )
    predictions = np.array(
        [data_dict["pd_probs"] for data_dict in infer_out["data"]]
    )

    cmp = ClassificationMetricsPipeline()
    metrics = cmp.run(
        len(infer_out["categories"]),
        labels,
        predictions
    )

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    return metrics


def plot_metrics(class_name_list: List[str], metrics: Dict, save_folder: str):
    plot_all(
        class_name_list,
        metrics["ap_list"],
        metrics["refined_pr_curves"],
        metrics["pr_curves"],
        np.array(metrics["confusion"]),
        np.array(metrics["confusion_col_norm"]),
        np.array(metrics["confusion_row_norm"]),
        save_folder
    )