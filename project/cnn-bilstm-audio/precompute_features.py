import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HFAudioMFCCDataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="small")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="features_small_train.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=max(1, os.cpu_count() - 2))

    args = parser.parse_args()

    target_classes = [
        "crowd-noise",
        "generator",
        "motorvehicle-horn",
        "mobile-music",
        "community-radio",
        "construction-site",
        "motorvehicle-siren",
        "car-alarm"
    ]

    dataset = HFAudioMFCCDataset(
        config_name=args.config,
        split=args.split,
        target_classes=target_classes
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )

    all_features = []
    all_labels = []

    for features, labels in tqdm(loader):
        all_features.append(features)
        all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    output_path = Path(args.output)

    torch.save(
        {
            "features": features,
            "labels": labels,
            "label_names": dataset.label_names,
            "class_to_id": dataset.class_to_id,
            "id_to_class": dataset.id_to_class,
            "config": args.config,
            "split": args.split
        },
        output_path
    )

    print("Saved:", output_path)
    print("Features:", features.shape)
    print("Labels:", labels.shape)
    print("Classes:", dataset.label_names)


if __name__ == "__main__":
    main()