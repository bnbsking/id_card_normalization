import timm
import torch


class TimmProbingModel(torch.nn.Module):
    def __init__(self, name: str, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            name,
            pretrained=True,
            num_classes=0,
        )
        self.backbone.requires_grad_(False)
        self.classifier = torch.nn.Linear(
            self.backbone.num_features,
            num_classes
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)
        return self.classifier(features)
    