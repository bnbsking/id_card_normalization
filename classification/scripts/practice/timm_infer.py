import torch
import timm
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model(
    "vit_base_patch14_dinov2.lvd142m",
    pretrained=True,
    num_classes=0,
)
model = model.to(device)
model.eval()


data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(
    **data_config,
    is_training=False,
)


image = Image.open("/dataset/0914/台胞證/20569495559749.jpg").convert("RGB")
x = transform(image).unsqueeze(0).to(device)
with torch.no_grad():
    feature = model(x)
print(feature.shape)  # (1,768)
