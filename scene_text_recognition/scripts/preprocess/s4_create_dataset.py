from tools.create_lmdb_dataset import main_custom


if __name__ == "__main__":
    exp_name = "exp0"

    main_custom(
        data_dir=f'/dataset/images/text_recognition/{exp_name}/train',
        save_folder=f"/dataset/images/text_recognition/{exp_name}/train_lmdb"
    )
    main_custom(
        data_dir=f'/dataset/images/text_recognition/{exp_name}/valid',
        save_folder=f"/dataset/images/text_recognition/{exp_name}/valid_lmdb"
    )
