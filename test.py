import torch

from train import Iris_Dataset

dataset = Iris_Dataset('test.data')
model = torch.load('weights.pt')
model.eval()
with torch.no_grad():
    output = model(dataset)
    _, pred = torch.max(output.data, 1)
    print(pred)