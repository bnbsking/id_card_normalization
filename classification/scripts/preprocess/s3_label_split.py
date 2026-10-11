from src.preprocess.label_split import split


if __name__ == "__main__":
    exp_name = "exp2"

    split(
        f"/dataset/labels/classification/{exp_name}/labeled.json",
        f"/dataset/labels/classification_transform/{exp_name}"
    )
    