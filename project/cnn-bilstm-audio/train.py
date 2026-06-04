import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from features_from_dataset import PrecomputedFeatureDataset
from model import CNNBiLSTMAudioClassifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(dataset, batch_size, val_ratio, test_ratio, seed, num_workers):
    labels = dataset.labels.numpy()
    indices = np.arange(len(labels))

    first_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_ratio,
        random_state=seed
    )

    train_val_idx, test_idx = next(
        first_splitter.split(indices, labels)
    )

    train_val_labels = labels[train_val_idx]

    val_size_adjusted = val_ratio / (1.0 - test_ratio)

    second_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size_adjusted,
        random_state=seed
    )

    train_idx_relative, val_idx_relative = next(
        second_splitter.split(train_val_idx, train_val_labels)
    )

    train_idx = train_val_idx[train_idx_relative]
    val_idx = train_val_idx[val_idx_relative]

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    split_indices = {
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist()
    }

    return train_loader, val_loader, test_loader, split_indices


def get_class_weights(labels, num_classes, device):
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    return weights.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for features, labels in tqdm(loader, desc="train", leave=False):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(features)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * features.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for features, labels in tqdm(loader, desc="val", leave=False):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(features)
        loss = criterion(logits, labels)

        total_loss += loss.item() * features.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, f1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--features", default="features_small_train.pt")
    parser.add_argument("--output", default="best_cnn_bilstm.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--test_ratio", type=float, default=0.15)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    dataset = PrecomputedFeatureDataset(args.features)

    num_classes = len(dataset.label_names)

    train_loader, val_loader, test_loader, split_indices = make_loaders(
        dataset=dataset,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        num_workers=args.num_workers
    )

    model = CNNBiLSTMAudioClassifier(num_classes=num_classes).to(device)

    if args.use_class_weights:
        weights = get_class_weights(dataset.labels, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_f1 = -1.0
    output_path = Path(args.output)

    print("Train samples:", len(train_loader.dataset))
    print("Val samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp
        )

        val_loss, val_acc, val_f1 = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_names": dataset.label_names,
                    "class_to_id": dataset.class_to_id,
                    "id_to_class": dataset.id_to_class,
                    "num_classes": num_classes,
                    "n_mfcc": 40,
                    "max_frames": dataset.features.shape[1],
                    "best_val_f1": best_val_f1,
                    "split_indices": split_indices,
                    "args": vars(args)
                },
                output_path
            )

            print("Saved:", output_path)

    print("Best val macro F1:", best_val_f1)


if __name__ == "__main__":
    main()