from src.train.show_history import plot


if __name__ == "__main__":
    exp_name = "exp0"
    plot(f"/app/outputs/training/pair_detection/{exp_name}/metrics.csv")
    