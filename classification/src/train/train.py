import json
import os
from typing import Dict, List

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.dataset.vertical_flip import VerticalFlipClassificationDataset
from src.models.probing import TimmProbingModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_annotations(path: str, num_classes: int) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        annotation = json.load(f)
        assert len(annotation["categories"]) == num_classes
        return annotation["data"]


def get_loaders(data_list: List[Dict], img_size: int, batch_size: int, shuffle: bool):
    dataset = VerticalFlipClassificationDataset(
        data_list=data_list,
        img_size=img_size
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
    )
    return loader


def preprare_model_criterion_optimizer(model_name: str, num_classes: int, lr: float):
    model = TimmProbingModel(
        model_name,
        num_classes=num_classes
    )
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        #weight_decay=1e-4,
    )
    return model, criterion, optimizer


def main(
        ant_path_train: str,
        ant_path_valid: str,
        img_size: int,
        batch_size: int,
        model_name: str,
        num_classes: int,
        epochs: int,
        lr: float,
        save_model_path: str,
        save_history_path: str,
    ):
    train_data_list = get_annotations(ant_path_train, num_classes)
    valid_data_list = get_annotations(ant_path_valid, num_classes)

    train_loader = get_loaders(train_data_list, img_size, batch_size, True)
    val_loader = get_loaders(valid_data_list, img_size, batch_size, False)

    model, criterion, optimizer = preprare_model_criterion_optimizer(
        model_name,
        num_classes,
        lr
    )

    history = {
        "train_loss_list": [],
        "valid_loss_list": [],
        "train_accuracy": [],
        "valid_accuracy": [],
        "all_accuracy": []
    }
    n_train = len(train_loader.dataset)
    n_valid = len(val_loader.dataset)
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_history_path), exist_ok=True)
    for epoch in tqdm(range(epochs)):
        # training
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            #
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            #
            train_loss += loss.item() * x.size(0) / n_train
            train_accuracy += (pred.argmax(dim=1) == y).sum().item() / n_train

        # validation
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0) / n_valid
                val_accuracy += (pred.argmax(dim=1) == y).sum().item() / n_valid

        history["train_loss_list"].append(train_loss)
        history["valid_loss_list"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["valid_accuracy"].append(val_accuracy)
        all_accuracy = (train_accuracy * n_train + val_accuracy * n_valid) / (n_train + n_valid)
        history["all_accuracy"].append(all_accuracy)

        with open(save_history_path, "w") as f:
            json.dump(history, f, indent=4)
        if all_accuracy >= history.get("best_all_acc", 0):
            history["best_all_acc"] = all_accuracy
            torch.save(model.state_dict(), save_model_path)
