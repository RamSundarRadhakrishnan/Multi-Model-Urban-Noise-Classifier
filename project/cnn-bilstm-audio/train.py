import argparse
import random
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

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


def apply_feature_augmentation(
    features,
    aug_prob=0.5,
    noise_std=0.01,
    time_mask_max=20,
    freq_mask_max=12
):
    if aug_prob <= 0:
        return features

    x = features.clone()

    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std

    batch_size, time_steps, feature_dim = x.shape

    for i in range(batch_size):
        if torch.rand(1, device=x.device).item() > aug_prob:
            continue

        if time_mask_max > 0:
            mask_len = torch.randint(1, min(time_mask_max, time_steps) + 1, (1,), device=x.device).item()
            start = torch.randint(0, time_steps - mask_len + 1, (1,), device=x.device).item()
            x[i, start:start + mask_len, :] = 0

        if freq_mask_max > 0:
            mask_len = torch.randint(1, min(freq_mask_max, feature_dim) + 1, (1,), device=x.device).item()
            start = torch.randint(0, feature_dim - mask_len + 1, (1,), device=x.device).item()
            x[i, :, start:start + mask_len] = 0

    return x

def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp, augment_features=False, aug_prob=0.5, noise_std=0.01, time_mask_max=20, freq_mask_max=12):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for features, labels in tqdm(loader, desc="train", leave=False):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if augment_features:
            features = apply_feature_augmentation(
                features,
                aug_prob=aug_prob,
                noise_std=noise_std,
                time_mask_max=time_mask_max,
                freq_mask_max=freq_mask_max
            )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
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


def save_training_curves(history, csv_path, plot_path):
    df = pd.DataFrame(history)
    df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(df["epoch"], df["train_loss"], label="Training loss")
    axes[0].plot(df["epoch"], df["val_loss"], label="Validation loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("CNN-BiLSTM training and validation loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df["epoch"], df["train_acc"], label="Training accuracy")
    axes[1].plot(df["epoch"], df["val_acc"], label="Validation accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("CNN-BiLSTM training and validation accuracy")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(df["epoch"], df["train_f1"], label="Training macro F1")
    axes[2].plot(df["epoch"], df["val_f1"], label="Validation macro F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1")
    axes[2].set_title("CNN-BiLSTM training and validation macro F1")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--features", default="features_small_train.pt")
    parser.add_argument("--output", default="best_cnn_bilstm.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--history_csv", default="training_history.csv")
    parser.add_argument("--plot_path", default="training_curves.png")
    parser.add_argument("--augment_features", action="store_true")
    parser.add_argument("--aug_prob", type=float, default=0.5)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--time_mask_max", type=int, default=20)
    parser.add_argument("--freq_mask_max", type=int, default=12)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    dataset = PrecomputedFeatureDataset(args.features)
    print("Feature tensor:", dataset.features.shape)
    print("Label tensor:", dataset.labels.shape)

    num_classes = len(dataset.label_names)

    train_loader, val_loader, test_loader, split_indices = make_loaders(
        dataset=dataset,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    input_dim = dataset.features.shape[2]
    model = CNNBiLSTMAudioClassifier(num_classes=num_classes, n_mfcc=input_dim).to(device)

    if args.use_class_weights:
        train_labels = dataset.labels[split_indices["train_idx"]]
        weights = get_class_weights(train_labels, num_classes, device)
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
        mode="max",
        factor=0.5,
        patience=4
    )

    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_f1 = -1.0
    output_path = Path(args.output)

    print("Train samples:", len(train_loader.dataset))
    print("Val samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    epochs_without_improvement = 0

    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            augment_features=args.augment_features,
            aug_prob=args.aug_prob,
            noise_std=args.noise_std,
            time_mask_max=args.time_mask_max,
            freq_mask_max=args.freq_mask_max
        )

        val_loss, val_acc, val_f1 = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        scheduler.step(val_f1)
        improved = val_f1 > best_val_f1 + args.min_delta

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "lr": optimizer.param_groups[0]["lr"]
            }
        )

        save_training_curves(
            history=history,
            csv_path=args.history_csv,
            plot_path=args.plot_path
        )

        if improved:
            best_val_f1 = val_f1
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_names": dataset.label_names,
                    "class_to_id": dataset.class_to_id,
                    "id_to_class": dataset.id_to_class,
                    "num_classes": num_classes,
                    "n_mfcc": input_dim,
                    "feature_dim" : input_dim,
                    "max_frames": dataset.features.shape[1],
                    "best_val_f1": best_val_f1,
                    "split_indices": split_indices,
                    "args": vars(args)
                },
            output_path
            )

            print("Saved:", output_path)
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{args.patience} epochs")

        if epochs_without_improvement >= args.patience:
            print("Early stopping triggered")
            break

    print("Best val macro F1:", best_val_f1)


if __name__ == "__main__":
    main()