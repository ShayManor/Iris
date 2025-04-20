import random

with open('IRIS.csv', 'r') as f:
    lines = f.readlines()
train = []
test = []
for line in lines:
    if random.Random().random() < 0.15:
        test.append(line)
    else:
        train.append(line)
with open('test.csv', 'w') as f:
    f.writelines(test)
with open('train.csv', 'w') as f:
    f.writelines(train)