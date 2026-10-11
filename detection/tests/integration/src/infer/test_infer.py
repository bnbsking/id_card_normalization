from src.infer.infer import DetectionInference, PairDetectionInference


class TestDetectionInference:
    def test_inference(self):
        img_path = "/dataset/images/remove_exif/0914/台胞證/20569495559749.jpg"
        pretrain_weights="/app/outputs/training/detection/exp2/checkpoint_best_ema.pth"
        threshold = 0.5

        obj = DetectionInference(pretrain_weights, threshold)
        result = obj.inference(img_path)
        print(result)
        """
        {'xmin': 63, 'ymin': 1383, 'xmax': 2793, 'ymax': 3131, 'confidence': 0.948}
        """


class TestPairDetectionInference:
    def test_inference(self):
        img_path = "/dataset/images/crop_vflip/0914/台胞證/20569495559749.jpg"
        pretrain_weights="/app/outputs/training/pair_detection/exp0/checkpoint_best_ema.pth"
        threshold = 0.5

        obj = PairDetectionInference(pretrain_weights, threshold)
        result = obj.inference(img_path)
        print(result)
        """
        {
            'boxes': [
                [910, 1421, 1521, 1527],
                [905, 777, 1394, 868],
                [895, 999, 1393, 1084],
                [1902, 785, 1986, 884],
                [1476, 999, 1979, 1089],
                [900, 420, 1280, 530]
            ],
            'confidence': [0.968, 0.94, 0.94, 0.939, 0.936, 0.931],
            'class_id': [3, 2, 2, 1, 2, 0]
        }
        """


if __name__ == "__main__":
    TestDetectionInference().test_inference()
    TestPairDetectionInference().test_inference()
