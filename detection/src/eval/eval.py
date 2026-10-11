import copy
import json
import os
from typing import Dict, List

from cosmodules.detection.metrics.export_worst import export_worst_cases
from cosmodules.detection.metrics.metrics_pipeline import DetectionMetricsPipeline
from cosmodules.detection.metrics.plot_all import plot_all
import numpy as np
from rfdetr import RFDETRMedium


def inference(
        ant_path: str,
        pretrain_weights: str,
        save_path: str
    ):
    with open(ant_path, "r") as f:
        annotations = json.load(f)
    num_classes = len(annotations["categories"])
    out = copy.deepcopy(annotations)

    model = RFDETRMedium(pretrain_weights=pretrain_weights)
    for i, data_dict in enumerate(annotations["data"]):
        img_path = data_dict["img_path"]
        detections = model.predict(img_path)

        out["data"][i]["pd_boxes"] = detections.xyxy.astype(np.int64).tolist()
        pd_probs = []  # shape=(boxes, num_classes)
        for confidence, pred_cls in zip(detections.confidence, detections.class_id):
            pd_probs.append(
                [float(confidence) if cid == pred_cls + 1 else (1 - float(confidence)) / (num_classes - 1) for cid in range(num_classes)]
            )
        out["data"][i]["pd_probs"] = pd_probs

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)


def full_metrics_pipeline(pred_path: str, save_path: str) -> Dict:
    with open(pred_path, "r", encoding="utf-8") as f:
        infer_out = json.load(f)

    labels = []  # len=images, each_shape=(num_boxes, 5), where (gt_cls, xmin, ymin, xmax, ymax)
    predictions = []  # len=images, each_shape=(num_boxes, 6), where (xmin, ymin, xmax, ymax, conf, pd_cls)
    for data_dict in infer_out["data"]:
        label = []
        for gt_cls, gt_box in zip(data_dict["gt_cls"], data_dict["gt_boxes"]):
            label.append([gt_cls] + gt_box)
        labels.append(np.array(label))

        pred = []
        for pd_box, pd_probs in zip(data_dict["pd_boxes"], data_dict["pd_probs"]):
            pred.append(pd_box + [max(pd_probs), max(list(range(len(pd_probs))), key=lambda x: pd_probs[x])])
        predictions.append(np.array(pred))
    
    dmp = DetectionMetricsPipeline()
    metrics = dmp.run(len(infer_out["categories"]), labels, predictions)
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