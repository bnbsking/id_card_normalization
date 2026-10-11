from src.infer.infer import single_inference_threshold, joint_inference_threshold


def test_single_inference_threshold():
    result = single_inference_threshold(
        model_path = "/models/svtrv2_small363_smtr_gtc_rctc_ch/pretrain/best.pth",
        infer_img = "/dataset/images/pair_crop/0914/身份證/26452494799045_address_0.jpg"
    )
    print(result)  # 河南省封丘县城关乡刘村


def test_joint_inference_threshold():
    result = joint_inference_threshold(
        model_path1 = "/models/svtrv2_small363_smtr_gtc_rctc_ch/pretrain/best.pth",
        model_path2 = "/app/outputs/finetune/exp0/best.pth",
        infer_img = "/dataset/images/pair_crop/0914/駕駛證/20118332944453_name_0.jpg"
    )
    print(result)  # 袁乾伟


if __name__ == "__main__":
    test_single_inference_threshold()
    test_joint_inference_threshold()
