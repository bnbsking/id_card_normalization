import json

from src.infer.infer import VerticalFlipInference


class TestVerticalFlipInference:
    def test_inference(self):
        exp_name = "exp2"
        img_path = "/dataset/images/crop/0914/台胞證/20569495559749.jpg"

        pretrained_weights = f"/models/vertical_flip_classification/model_{exp_name}.pth"
        metrics_path = f"/app/outputs/{exp_name}/eval/metrics.json"
        
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            threshold = metrics["best_threshold"]
        infer = VerticalFlipInference(pretrained_weights, threshold)
        result = infer.inference(img_path)
        print(result)  # {'pd_cls': 0, 'vflip_confidence': 0.12082499265670776}
        

if __name__ == "__main__":
    obj = TestVerticalFlipInference()
    obj.test_inference()
