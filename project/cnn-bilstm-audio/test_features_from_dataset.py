from features_from_dataset import PrecomputedFeatureDataset


dataset = PrecomputedFeatureDataset("features_small_train.pt")

x, y = dataset[0]

print(x.shape)
print(y)
print(dataset.label_names)
print(len(dataset))