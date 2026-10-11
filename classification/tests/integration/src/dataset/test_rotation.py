import os
from pathlib import Path

from PIL import Image
import torch

from src.dataset.rotation import RotationClassificationDataset


class TestRotationClassificationDataset:
    def save_rotation_samples(
            self,
            dataset: RotationClassificationDataset,
            output_dir: str
        ):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for img_idx in range(len(dataset) // 4):  # len(dataset) // 4 = len(images)
            for i in range(4):
                dataset_idx = img_idx * 4 + i

                x, y = dataset[dataset_idx]
                x = x.clone()

                mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype).view(3, 1, 1)

                x = x * std + mean
                x = x.clamp(0, 1)
                x = (x * 255).byte()
                x = x.permute(1, 2, 0).cpu().numpy()

                img = Image.fromarray(x)
                label = y.item()

                filename = (
                    f"{os.path.basename(dataset.img_path_list[dataset_idx]).split(".")[0]}"
                    f"_label_{label}.jpg"
                )
                img.save(output_path / filename)

        print(f"Saved {len(dataset)} images to: {output_path}")

    def test_run(self):
        trainset = RotationClassificationDataset(
            data_list=[
                # {
                #     "data_path": "/dataset/0914/台胞證/20569495559749.jpg",
                #     "gt_cls": 0
                # },
                # {
                #     "data_path": "/dataset/0914/台胞證/23834295518533.jpg",
                #     "gt_cls": 0
                # },
                # {
                #     "data_path": "/dataset/0914/台胞證/24297395536837.jpg",
                #     "gt_cls": 0
                # },
                # {
                #     "data_path": "/dataset/0914/台胞證/25305371872581.jpg",
                #     "gt_cls": 0
                # },
                # {
                #     "data_path": "/dataset/0914/台胞證/25391866699461.jpg",
                #     "gt_cls": 0
                # },
                # {
                #     "data_path": "/dataset/0914/台胞證/25878144816965.jpg",
                #     "gt_cls": 0
                # },
                {
                    "data_path": "/dataset/0914/台胞證/25900262245701.jpg",
                    "gt_cls": 3
                },
                # {
                #     "data_path": "/dataset/0914/台胞證/25900406125253.jpg",
                #     "gt_cls": 0
                # },
                # {
                #     "data_path": "/dataset/0914/台胞證/26331023355845.jpg",
                #     "gt_cls": 3
                # },
                # {
                #     "data_path": "/dataset/0914/台胞證/26364203371077.jpg",
                #     "gt_cls": 0
                # }
            ],
            img_size=518,
            mode="train"
        )
        self.save_rotation_samples(
            trainset,
            "/app/tests/integration/output/dataset/rotation"
        )


if __name__ == "__main__":
    test = TestRotationClassificationDataset()
    test.test_run()
