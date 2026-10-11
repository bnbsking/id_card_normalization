import glob
import json
import os

from src.eval.eval import inference, full_metrics_pipeline, plot_metrics


if __name__ == "__main__":
    exp_name = "exp0"

    ant_path = f"/dataset/labels/pair_detection_transform/{exp_name}/.others/general.json"  # general_valid.json
    model_folder = f"/app/outputs/training/pair_detection/{exp_name}"

    best_map = 0
    for model_path in glob.glob(f"{model_folder}/*.ckpt") + glob.glob(f"{model_folder}/*.pth"):
        # inference
        save_folder = os.path.join(model_folder, "eval", os.path.basename(model_path).split(".")[0])
        save_prediction_path = os.path.join(save_folder, "prediction.json")
        inference(ant_path, model_path, save_prediction_path)

        # compute metrics
        save_metrics_path = os.path.join(save_folder, "metrics.json")
        metrics = full_metrics_pipeline(save_prediction_path, save_metrics_path)
        plot_metrics(
            json.load(open(ant_path, "r", encoding="utf-8"))["categories"][1:],
            metrics,
            save_folder
        )
        if metrics["map"] >= best_map:
            with open(os.path.join(model_folder, "best_model_path.txt"), "w", encoding="utf-8") as f:
                f.write(model_path)
            best_map = metrics["map"]
