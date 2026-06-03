import torch
from model import CNNBiLSTMAudioClassifier


num_classes = 8
batch_size = 4
time_steps = 174
n_mfcc = 40

model = CNNBiLSTMAudioClassifier(num_classes=num_classes)

x = torch.randn(batch_size, time_steps, n_mfcc)
logits = model(x)

print(logits.shape)