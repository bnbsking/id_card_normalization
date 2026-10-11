import os
import yaml

from src.train.train import main


if __name__ == "__main__":
    with open(os.environ["BASE_CFG_PATH"], "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["train"]
    main(
        ant_path_train=cfg["dataset"]["ant_path_train"],
        ant_path_valid=cfg["dataset"]["ant_path_valid"],
        img_size=cfg["dataset"]["img_size"],
        batch_size=cfg["dataset"]["batch_size"],
        model_name=cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        epochs=cfg["model"]["epochs"],
        lr=cfg["model"]["lr"],
        save_model_path=cfg["model"]["save_model_path"],
        save_history_path=cfg["model"]["save_history_path"]
    )
