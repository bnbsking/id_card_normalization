from rfdetr import RFDETRMedium


def train(dataset_dir: str, output_dir: str):
    model = RFDETRMedium()
    model.train(
        dataset_dir=dataset_dir,
        epochs=100,
        batch_size="auto",
        lr=1e-4,
        output_dir=output_dir,
    )
