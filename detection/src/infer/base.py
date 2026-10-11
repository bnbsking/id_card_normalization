from rfdetr import RFDETRMedium


class BaseInference:
    def __init__(self, pretrain_weights: str, threshold: float):
        self.pretrain_weights = None
        self.threshold = None
        self.model = None
        self.set_cfg(pretrain_weights, threshold)
        
    def set_cfg(self, pretrain_weights: str | None = None, threshold: float | None = None):
        if pretrain_weights is not None and pretrain_weights != self.pretrain_weights:
            self.pretrain_weights = pretrain_weights
            self.model = RFDETRMedium(pretrain_weights=pretrain_weights)
        if threshold is not None:
            self.threshold = threshold

    def inference(self, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")
    