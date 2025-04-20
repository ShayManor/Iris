import torch
from torch import nn
from torch.utils.data import DataLoader

from train import Iris_Dataset


def check_correct(given: list[float], exp: list[float]):
    max_g = max(given)
    mag_e = max(exp)
    return given.index(max_g) == exp.index(mag_e)


dataset = Iris_Dataset('test.csv')
model = torch.load('weights.pt', weights_only=False)
model.eval()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
criterion = nn.MSELoss()

correct_counter, total = 0, 0
test_loss = 0
for data, label in dataloader:
    output = model(data)
    predicted = output.data
    correct: bool = check_correct(predicted.tolist(), label[0].tolist())
    if not correct:
        print(f"Expected: {label.tolist()}, Predicted: {predicted.tolist()}")
    correct_counter += int(correct)
    total += label.size(0)
    loss = criterion(output, label)
    test_loss += loss.item() * data.size(0)
print(f'Testing Loss:{test_loss / len(dataloader)}')
print(f'Correct Predictions: {correct_counter}/{total}')
