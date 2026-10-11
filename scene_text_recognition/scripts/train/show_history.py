import os
import re

import matplotlib.pyplot as plt


def get_latest_training_log(history: str):
    with open(history, "r", encoding="utf-8") as f:
        logs = f.read()
    matches = list(re.finditer(r"epoch: \[1/100\]", logs))
    if matches:
        last_match = matches[-1]
        return logs[last_match.start():]
    else:
        return ""


def plot(logs: str, save_path: str):
    loss = [float(x) for x in re.findall(r", loss: ([\d.]+)", logs)]
    train_loss = loss[::2]
    valid_loss = loss[1::2]

    plt.plot(range(len(train_loss)), train_loss, label="Train Loss")
    plt.plot(range(len(valid_loss)), valid_loss, label="Valid Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    exp_name = "exp0"

    log_path = f"/app/outputs/finetune/{exp_name}/train.log"
    logs = get_latest_training_log(log_path)
    plot(logs, os.path.join(os.path.dirname(log_path), "training_loss.png"))
