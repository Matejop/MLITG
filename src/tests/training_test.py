from MLL.model import Model
from copy import deepcopy
import random as rnd

if __name__ == "__main__":
    training_x = []
    training_y = []
    with open("c:/Users/matej/source/MLITG/tests/MNIST/mnist_train.csv") as f:
        file = f.read()
        file_lines = file.split("\n")
        file_lines.pop()
        for line in file_lines:
            split_line = line.split(",")
            training_x.append([float(pixel) / 255 for pixel in split_line[1:]])
            label = split_line[0]
            y = [0 for _ in range(10)]
            y[int(label)] = 1                
            training_y.append(y)

    testing_x = []
    testing_y = []
    with open("c:/Users/matej/source/MLITG/src/MNIST_CSV/mnist_test.csv") as f:
        file = f.read()
        file_lines = file.split("\n")
        file_lines.pop()
        for line in file_lines:
            split_line = line.split(",")
            testing_x.append([float(pixel) / 255 for pixel in split_line[1:]])
            label = split_line[0]
            y = [0 for _ in range(10)]
            y[int(label)] = 1                
            testing_y.append(y)
    model = Model.class_fit(training_x, training_y, [784, 256, 10])
    print(f"Accuracy:{model.measure_accuracy(testing_x, testing_y)}")
    model.save("c:/Users/matej/source/MLITG/models/")