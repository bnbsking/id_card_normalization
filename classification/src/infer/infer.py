from typing import Dict

import torch

from src.dataset.vertical_flip import VerticalFlipClassificationDataset
from src.infer.base import BaseInference


class VerticalFlipInference(BaseInference):
    def inference(self, img_path: str) -> Dict[str, int | float]:
        data_list = [
            {
                "data_path": img_path,
                "gt_cls": -1
            }
        ]
        dataset = VerticalFlipClassificationDataset(data_list, self.img_size)
        x, _ = dataset[0]
        x = x.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(x)
        pd_probs = torch.softmax(pred, dim=1)
        pd_probs = pd_probs.detach().cpu().numpy().tolist()[0]
                
        confidence = pd_probs[1]
        return {
            "pd_cls": int(confidence >= self.threshold),
            "vflip_confidence": confidence
        }
    