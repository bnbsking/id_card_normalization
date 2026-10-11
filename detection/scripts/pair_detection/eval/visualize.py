import os

from src.eval.visualize import run_export_worst_cases, vis


if __name__ == "__main__":
    exp_name = "exp0"

    with open(f"/app/outputs/training/pair_detection/{exp_name}/best_model_path.txt", "r") as f:
        text = f.read().strip()
        eval_folder = os.path.join(
            f"/app/outputs/training/pair_detection/{exp_name}/eval",
            os.path.basename(text).split(".")[0]
        )

    run_export_worst_cases(
        pred_path=os.path.join(eval_folder, "prediction.json"),
        metrics_path=os.path.join(eval_folder, "metrics.json"),
        save_folder=os.path.join(eval_folder, "export_worst")
    )

    if 1:
        vis(
            pred_path=os.path.join(eval_folder, "prediction.json"),
            save_folder=os.path.join(eval_folder, "vis")
        )
