import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from feature_dataset import PrecomputedFeatureDataset
from model import CNNBiLSTMAudioClassifier


def get_eval_indices(labels, test_ratio, seed):
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_ratio,
        random_state=seed
    )

    _, test_idx = next(splitter.split(torch.zeros(len(labels)), labels))
    return test_idx


@torch.no_grad()
def run_eval(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    for features, labels in tqdm(loader, desc="evaluate"):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(features)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        total_loss += loss.item() * features.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, all_labels, all_preds, all_probs


def plot_confusion_matrix(cm, label_names, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm)

    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("CNN-BiLSTM Confusion Matrix")

    for i in range(len(label_names)):
        for j in range(len(label_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--features", default="features_large_train.pt")
    parser.add_argument("--checkpoint", default="best_cnn_bilstm.pt")
    parser.add_argument("--output_dir", default="metrics_cnn_bilstm")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PrecomputedFeatureDataset(args.features)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    label_names = checkpoint.get("label_names", dataset.label_names)
    num_classes = len(label_names)

    test_idx = get_eval_indices(dataset.labels.numpy(), args.test_ratio, args.seed)
    test_dataset = Subset(dataset, test_idx)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = CNNBiLSTMAudioClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()

    test_loss, y_true, y_pred, y_prob = run_eval(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device
    )

    accuracy = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "test_loss": test_loss,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "num_test_samples": len(test_dataset),
        "classes": label_names
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")

    pd.DataFrame(
        cm,
        index=label_names,
        columns=label_names
    ).to_csv(output_dir / "confusion_matrix.csv")

    plot_confusion_matrix(
        cm=cm,
        label_names=label_names,
        output_path=output_dir / "confusion_matrix.png"
    )

    pred_df = pd.DataFrame({
        "actual_id": y_true,
        "predicted_id": y_pred,
        "actual_label": [label_names[i] for i in y_true],
        "predicted_label": [label_names[i] for i in y_pred]
    })

    for i, label in enumerate(label_names):
        pred_df[f"prob_{label}"] = [row[i] for row in y_prob]

    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    print(json.dumps(metrics, indent=4))
    print("Saved metrics to:", output_dir)


if __name__ == "__main__":
    main()