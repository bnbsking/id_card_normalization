import json
from typing import Dict

import numpy as np

from .base import BaseInference


class DetectionInference(BaseInference):
    def inference(self, img_path: str) -> Dict[str, int]:
        detections = self.model.predict(img_path)
        if detections.confidence.size == 0:
            return {}
        
        max_conf_id = np.argmax(detections.confidence).item()
        confidence = round(detections.confidence[max_conf_id].item(), 3)
        if confidence < self.threshold:
            return {}

        xmin, ymin, xmax, ymax = detections.xyxy.astype(np.int64)[max_conf_id].tolist()
        return {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "confidence": confidence
        }


class PairDetectionInference(BaseInference):
    def inference(self, img_path: str) -> Dict[str, int]:
        detections = self.model.predict(img_path)
        if detections.confidence.size == 0:
            return {}

        out = {
            "boxes": detections.xyxy.astype(np.int64).tolist(),
            "confidence": [round(conf, 3) for conf in detections.confidence.astype(np.float32).tolist()],
            "class_id": detections.class_id.astype(np.int64).tolist()
        }
        positive_indices = [i for i, conf in enumerate(out["confidence"]) if conf >= self.threshold]
        for key in out:
            out[key] = [out[key][idx] for idx in positive_indices]
        return out