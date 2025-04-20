import os
import torch
from torch import nn, optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pandas as pd

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

indices = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2,
}


def apply_indicies(species):
    return indices[species]


class Iris_Dataset(Dataset):
    def __init__(self, path):
        self.df = pd.read_csv(path, dtype={'species': str, 'sepal_length': float, 'sepal_width': float, 'petal_length': float, 'petal_width': float})
        self.df['species_f'] = self.df['species'].apply(apply_indicies)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        t = torch.tensor(
            [self.df['sepal_length'][idx], self.df['sepal_width'][idx], self.df['petal_length'][idx],
             self.df['petal_width'][idx]], dtype=torch.float32)
        ret = torch.zeros(3)
        ret[self.df['species_f'][idx]] += 1
        return t, ret


class NeuralNetwork(nn.Module):
    # Input: 4, output, 1
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.r2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = x[0]
        x = self.fc1(x)
        x = self.r1(x)
        x = self.fc2(x)
        x = self.r2(x)
        x = self.fc3(x)
        return x


model = NeuralNetwork()
dataset = Iris_Dataset('train.csv')
dataloader = DataLoader(dataset, batch_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
epochs = 40
for epoch in range(epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
torch.save(model, 'weights.pt')