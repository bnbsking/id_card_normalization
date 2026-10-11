from src.train.train import train


if __name__ == "__main__":
    exp_name = "exp2"
    dataset_dir = f"/dataset/labels/detection_transform/{exp_name}"
    output_dir = f"/app/outputs/training/detection/{exp_name}"
    
    train(dataset_dir, output_dir) 
