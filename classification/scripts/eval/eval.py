import json
import os

from cosmodules.classification.metrics.export_worst import export_worst_cases
import yaml

from src.eval.eval import load_model, inference, full_metrics_pipeline, plot_metrics


if __name__ == "__main__":
    exp_name = "exp2"

    with open(os.environ["BASE_CFG_PATH"], "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["train"]
    ant_path = cfg["dataset"]["ant_path_all"]  # "ant_path_valid"
    output_folder = f"/app/outputs/{exp_name}"
    output_eval_folder = os.path.join(output_folder, "eval")
    save_prediction_path = os.path.join(output_eval_folder, "prediction.json")

    # inference
    model = load_model(
        cfg["model"]["name"],
        cfg["model"]["num_classes"],
        os.path.join("/models/vertical_flip_classification", f"model_{exp_name}.pth")
    )
    inference(
        ant_path,
        cfg["dataset"]["img_size"],
        model,
        save_prediction_path,
        do_flip=True
    )

    # metrics
    save_metrics_path = os.path.join(output_eval_folder, "metrics.json")
    metrics = full_metrics_pipeline(save_prediction_path, save_metrics_path)
    plot_metrics(
        json.load(open(ant_path, "r", encoding="utf-8"))["categories"],
        metrics,
        output_eval_folder
    )

    # export
    with open(ant_path, "r", encoding="utf-8") as f:
        ant = json.load(f)
    data_path_list = [data_dict["data_path"] for data_dict in ant["data"]]
    export_worst_cases(
        data_path_list,
        metrics["sorted_accuracy_index_list"],
        os.path.join(output_eval_folder, "export_worst")
    )
