from src.dataset.labeled_to_ready_train import visualize


if __name__ == "__main__":
    exp_name = "exp2"

    ant_path = f"/dataset/labels/detection_transform/{exp_name}/.others/general.json"
    output_dir = f"/app/outputs/show_labels/detection/{exp_name}"

    visualize(ant_path, output_dir)
