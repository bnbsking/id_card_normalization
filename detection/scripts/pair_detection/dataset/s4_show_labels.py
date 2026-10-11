from src.dataset.labeled_to_ready_train import visualize


if __name__ == "__main__":
    exp_name = "exp0"

    ant_path = f"/dataset/labels/pair_detection_transform/{exp_name}/.others/general.json"
    output_dir = f"/app/outputs/show_labels/pair_detection/{exp_name}"

    visualize(ant_path, output_dir)
