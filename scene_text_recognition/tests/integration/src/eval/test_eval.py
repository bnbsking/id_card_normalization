from src.eval.eval import (
    BaseEvaluation,
    SingleEvaluation,
    JointEvaluation
)


class TestBaseEvaluation:
    def test_metrics_export_worst(self):
        labels = {
            "data": [
                {"gt_text": "hello", "pd_text": "hello", "score": 1.0},
                {"gt_text": "world", "pd_text": "word", "score": 0.75},
                {"gt_text": "test", "pd_text": "tst", "score": 0.5}
            ]
        }
        metrics = BaseEvaluation.metrics_export_worst(labels, 10)
        print(metrics)
        """
        {
            'accuracy': 0.75,
            'worst_cases': [
                {'gt_text': 'test', 'pd_text': 'tst', 'score': 0.5},
                {'gt_text': 'world', 'pd_text': 'word', 'score': 0.75}
            ]
        }
        """
    

class TestSingleEvaluation:
    def test_infer_all(self):
        model_path = "/models/svtrv2_small363_smtr_gtc_rctc_ch/pretrain/best.pth"
        labels_path = "/dataset/labels/text_recognition/exp0/labels.json"
        output_path = "/app/outputs/pretrain/eval/prediction.json"        
        SingleEvaluation.infer_all(model_path, labels_path, output_path)


class TestJointEvaluation:
    def test_infer_all(self):
        model_path1 = "/models/svtrv2_small363_smtr_gtc_rctc_ch/pretrain/best.pth"
        model_path2 = "/app/outputs/finetune/exp0/best.pth"
        labels_path = "/dataset/labels/text_recognition/exp0/labels.json"
        output_path = "/app/outputs/pretrain/eval/prediction.json"        
        JointEvaluation.joint_infer_all(model_path1, model_path2, labels_path, output_path)


if __name__ == "__main__":
    obj = TestBaseEvaluation()
    obj.test_metrics_export_worst()
    
    obj = TestSingleEvaluation()
    obj.test_infer_all()

    obj = TestJointEvaluation()
    obj.test_infer_all()
    