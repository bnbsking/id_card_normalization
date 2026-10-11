from src.train.train import train


if __name__ == '__main__':
    exp_name = "exp0"
    
    train(
        output_dir=f'/app/outputs/finetune/{exp_name}',
        train_data_dir_list=[f'/dataset/images/text_recognition/{exp_name}/train_lmdb'],
        eval_data_dir_list=[f'/dataset/images/text_recognition/{exp_name}/valid_lmdb']
    )
