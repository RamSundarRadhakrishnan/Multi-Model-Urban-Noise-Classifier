from dataset import HFAudioMFCCDataset


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
    config_name="small",
    split="train",
    target_classes=target_classes
)

x, y = dataset[0]

print(x.shape)
print(y)
print(dataset.label_names)
print(dataset.class_to_id)
print(len(dataset))