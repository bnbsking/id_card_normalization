import copy
import os
import json
from typing import Dict

from rapidfuzz import fuzz

from src.infer.infer import single_inference_threshold, joint_inference_threshold


class BaseEvaluation:
    @staticmethod
    def add_accuracy(labels: Dict, save_path: str | None = None) -> Dict:
        for i, data_dict in enumerate(labels["data"]):
            pattern = data_dict["gt_text"]
            text = data_dict["pd_text"]
            score = fuzz.ratio(pattern, text) / 100
            labels["data"][i]["score"] = score

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(labels, f, ensure_ascii=False, indent=4)
        return labels

    @staticmethod
    def metrics_export_worst(labels: Dict, top_k: int, save_path: str | None = None) -> Dict:
        metrics = {}
        
        scores = [data_dict["score"] for data_dict in labels["data"]]
        metrics["accuracy"] = round(sum(scores) / (len(scores) + 1e-10), 3)
        
        inaccuracte_dicts = [data_dict for data_dict in labels["data"] if data_dict["score"] < 1.0]
        metrics["worst_cases"] = sorted(inaccuracte_dicts, key=lambda x: x["score"])[:top_k]
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=4)
        return metrics

    def run(self, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")


class SingleEvaluation(BaseEvaluation):
    @staticmethod
    def infer_all(model_path: str, labels_path: str, save_path: str | None = None) -> Dict:
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        
        for i, data_dict in enumerate(copy.deepcopy(labels)["data"]):
            result = single_inference_threshold(
                model_path=model_path,
                infer_img=data_dict["data_path"]
            )
            labels["data"][i]["pd_text"] = result

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(labels, f, ensure_ascii=False, indent=4)

        return labels

    def run(self, model_path: str, labels_path: str, output_folder: str | None = None) -> Dict:
        labels = self.infer_all(
            model_path=model_path,
            labels_path=labels_path,
            save_path=os.path.join(output_folder, "prediction.json")
        )
        self.add_accuracy(labels, os.path.join(output_folder, "prediction.json"))
        self.metrics_export_worst(labels, 10, os.path.join(output_folder, "metrics.json"))


class JointEvaluation(BaseEvaluation):
    @staticmethod
    def joint_infer_all(model_path1: str, model_path2: str, labels_path: str, save_path: str | None = None) -> Dict:
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        
        for i, data_dict in enumerate(copy.deepcopy(labels)["data"]):
            result = joint_inference_threshold(
                model_path1=model_path1,
                model_path2=model_path2,
                infer_img=data_dict["data_path"]
            )
            labels["data"][i]["pd_text"] = result

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(labels, f, ensure_ascii=False, indent=4)

        return labels

    def run(self, model_path1: str, model_path2: str, labels_path: str, output_folder: str | None = None) -> Dict:
        labels = self.joint_infer_all(
            model_path1=model_path1,
            model_path2=model_path2,
            labels_path=labels_path,
            save_path=os.path.join(output_folder, "prediction.json")
        )
        self.add_accuracy(labels, os.path.join(output_folder, "prediction.json"))
        self.metrics_export_worst(labels, 10, os.path.join(output_folder, "metrics.json"))
