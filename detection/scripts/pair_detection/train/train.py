from src.train.train import train


if __name__ == "__main__":
    exp_name = "exp0"
    dataset_dir = f"/dataset/labels/pair_detection_transform/{exp_name}"
    output_dir = f"/app/outputs/training/pair_detection/{exp_name}"

    train(dataset_dir, output_dir) 
