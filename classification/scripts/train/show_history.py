from src.train.show_history import plot


if __name__ == "__main__":
    exp_name = "exp2"

    plot(f"/app/outputs/{exp_name}/history.json")
