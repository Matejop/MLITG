from MLL.model import Model
from MLL.utils.trainer import Trainer
import orjson
import gzip

if __name__ == "__main__":
    data = orjson.loads(gzip.open("c:/Users/matej/source/MLITG/src/tests/MNIST/mnist_preprocessed.gz").read())
    #training_x = []
    #training_y = []
    #with open("c:/Users/matej/source/MLITG/src/tests/MNIST/mnist_train.csv") as f:
    #    file = f.read()
    #    file_lines = file.split("\n")
    #    file_lines.pop()
    #    for line in file_lines:
    #        split_line = line.split(",")
    #        training_x.append([float(pixel) / 255 for pixel in split_line[1:]])
    #        label = split_line[0]
    #        y = [0 for _ in range(10)]
    #        y[int(label)] = 1                
    #        training_y.append(y)

    #testing_x = []
    #testing_y = []
    #with open("c:/Users/matej/source/MLITG/src/tests/MNIST/mnist_test.csv") as f:
    #    file = f.read()
    #    file_lines = file.split("\n")
    #    file_lines.pop()
    #    for line in file_lines:
    #        split_line = line.split(",")
    #        testing_x.append([float(pixel) / 255 for pixel in split_line[1:]])
    #       label = split_line[0]
    #        y = [0 for _ in range(10)]
    #        y[int(label)] = 1                
    #        testing_y.append(y)
    trainer = Trainer(
        activation="relu",
        max_depth=3,
        trials=3,
        epochs=5,
        layer_size_start=256,
        layer_size_end=300,
        batch_size_start=1000,
        batch_size_end=1500,
        learning_rate_start=0.01,
        learning_rate_end=0.02,
    )
    model = trainer.random_search(data["training_x"], data["training_y"], data["validation_x"], data["validation_y"], 3)
    print(f"Accuracy:{model.measure_accuracy(data["testing_x"], data["testing_y"])}")
    model.save("c:/Users/matej/source/MLITG/models/")