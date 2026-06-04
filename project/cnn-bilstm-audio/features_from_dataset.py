import torch
from torch.utils.data import Dataset


class PrecomputedFeatureDataset(Dataset):
    def __init__(self, feature_path):
        data = torch.load(feature_path, map_location="cpu")

        self.features = data["features"].float()
        self.labels = data["labels"].long()

        self.label_names = data["label_names"]
        self.class_to_id = data["class_to_id"]
        self.id_to_class = data["id_to_class"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]