from src.eval.eval import SingleEvaluation, JointEvaluation


if __name__ == "__main__":
    exp_name = "exp0"

    # # custom pretrain
    # SingleEvaluation().run(
    #     model_path="/models/svtrv2_small363_smtr_gtc_rctc_ch/pretrain/best.pth",
    #     labels_path=f"/dataset/labels/text_recognition/{exp_name}/labels.json",
    #     output_folder=f"/app/outputs/pretrain/{exp_name}/eval"
    # )

    # custom finetune
    JointEvaluation().run(
        model_path1="/models/svtrv2_small363_smtr_gtc_rctc_ch/pretrain/best.pth",
        model_path2=f"/app/outputs/finetune/{exp_name}/best.pth",
        labels_path=f"/dataset/labels/text_recognition/{exp_name}/labels.json",
        output_folder=f"/app/outputs/finetune/{exp_name}/eval"
    )
