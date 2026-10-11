import os

import matplotlib.pyplot as plt
import pandas as pd


def plot(metrics_csv_path: str):
    df = pd.read_csv(metrics_csv_path)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df["train/loss"][df["train/loss"].notnull()].tolist(), label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df["val/mAP_50"][df["val/mAP_50"].notnull()].tolist(), label="Valid MAP")
    plt.xlabel("Epoch")
    plt.ylabel("mAP_50")
    plt.title("Valid MAP over Epochs")
    plt.legend()

    plt.savefig(os.path.join(os.path.dirname(metrics_csv_path), "metrics.png"))
    plt.close()
    