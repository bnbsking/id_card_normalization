import json
import os

import matplotlib.pyplot as plt


def plot(history_path: str):
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss_list"], label="Train Loss")
    plt.plot(history["valid_loss_list"], label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["valid_accuracy"], label="Valid Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()

    plt.savefig(os.path.join(os.path.dirname(history_path), "metrics.png"))
    plt.close()
