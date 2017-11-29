import numpy as np
import random

file = open("iris-dataset.csv", "r")
data = list()
for line in file:
    data.append(line.split(','))
    file.close()
    random.shuffle(data)
    train_data = data[:int((len(data) + 1) * .80)]  # Remaining 80% to training set
    test_data = data[int(len(data) * .80 + 1):]  # Splits 20% data to test set