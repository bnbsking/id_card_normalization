import os

from tools.infer_rec import main_custom
import yaml

from src.custom_cfg import recursive_merge


def inference(model_path: str, infer_img: str) -> str:
    update_cfg = {
        "Global": {
            "pretrained_model": model_path,
            "infer_img": infer_img,
        }
    }
    with open('/app/cfgs/infer.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg = recursive_merge(cfg, update_cfg)
    main_custom(cfg)

    output_path = os.path.join(os.environ["OPENOCR_FOLDER"], "rec_results/rec_results.txt")
    with open(output_path, "r", encoding="utf-8") as f:
        result_line = [ele for ele in f.readlines()[0].strip().split("\t") if ele.strip()]
    img_path = result_line[0]
    confidence = result_line[-1]
    return {"text": "".join(result_line[1:-1]), "confidence": confidence}


def single_inference_threshold(model_path: str, infer_img: str, threshold: float = 0.5) -> str:
    result = inference(model_path, infer_img)
    if result["confidence"] and float(result["confidence"]) < threshold:
        return ""
    else:
        return result["text"]


def joint_inference_threshold(model_path1: str, model_path2: str, infer_img: str, threshold: float = 0.5) -> str:
    result1 = inference(model_path1, infer_img)
    result2 = inference(model_path2, infer_img)
    breakpoint()
    if float(result1["confidence"]) < threshold and float(result2["confidence"]) < threshold:
        return ""
    return result1["text"] if float(result1["confidence"]) > float(result2["confidence"]) else result2["text"]
