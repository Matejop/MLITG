import numpy as np
import pickle
import gzip
import os
#TODO add filepaths to global config
def preprocess_data(filepath="./data/preprocessed/mnist_preprocessed.npz"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    training_data, validation_data, test_data = load_raw_data("./mnist_new.pkl.gz")
    training_answers = []
    validation_answers = []
    test_answers = []
    for i in range(len(training_data[0])):
        training_data[0][i] = training_data[0][i].flatten()
        training_answers.append(vectorized_result(training_data[1][i]))
    for i in range(len(validation_data[0])):
        validation_data[0][i] = validation_data[0][i].flatten()
        validation_answers.append(vectorized_result(validation_data[1][i]))
    for i in range(len(test_data[0])):         
        test_data[0][i] = test_data[0][i].flatten()
        test_answers.append(vectorized_result(test_data[1][i]))
    np.savez_compressed(filepath,
                        train_x=np.array(training_data[0]), train_y=np.array(training_answers),
                        val_x=np.array(validation_data[0]), val_y=np.array(validation_answers),
                        test_x=np.array(test_data[0]), test_y=np.array(test_answers))
    print(f"Data saved to {os.path.abspath(filepath)}")

def load_raw_data(filepath = "./data/raw/mnist.pkl.gz"):
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    with gzip.open(filepath, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        training_data, validation_data, test_data = u.load()
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

if __name__ == '__main__':
    preprocess_data()
