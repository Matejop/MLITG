import pickle
import orjson
import gzip
import os

#TODO add filepaths to global config

RAW_PATH = os.path.join("data", "raw", "mnist_new.pkl.gz")
PREPROCESSED_PATH = os.path.join("data", "preprocessed", "mnist_preprocessed.gz")


def preprocess_data(filepath = PREPROCESSED_PATH):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    training_data, validation_data, test_data = load_raw_data()
    training_input = []
    training_answers = []
    validation_input = []
    validation_answers = []
    test_input = []
    test_answers = []
    for i in range(len(training_data[0])):
        training_input.append(training_data[0][i].flatten().tolist())
        for j in range(len(training_input[-1])):
            training_input[-1][j] = float(training_input[-1][j])
        training_answers.append(vectorized_result(training_data[1][i]))
    for i in range(len(validation_data[0])):
        validation_input.append(validation_data[0][i].flatten().tolist())
        for j in range(len(validation_input[-1])):
            validation_input[-1][j] = float(validation_input[-1][j])
        validation_answers.append(vectorized_result(validation_data[1][i]))
    for i in range(len(test_data[0])):         
        test_input.append(test_data[0][i].flatten().tolist())
        for j in range(len(test_input[-1])):
            test_input[-1][j] = float(test_input[-1][j])
        test_answers.append(vectorized_result(test_data[1][i]))
    
    data_dict = {
        "training_x": training_input,
        "training_y": training_answers,
        "validation_x": validation_input,
        "validation_y": validation_answers,
        "testing_x": test_input,
        "testing_y": test_answers,
    }

    with gzip.open(filepath, "w") as f:
        f.write(orjson.dumps(
            data_dict, 
            option=orjson.OPT_INDENT_2)
        )
    print(f"Data saved to {os.path.abspath(filepath)}")

def load_raw_data(filepath = RAW_PATH):
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
    e = []
    for i in range(10):
        e.append(0.0 if i != j else 1.0)
    return e

if __name__ == "__main__":
    RAW_PATH = os.path.join("raw", "mnist_new.pkl.gz")
    PREPROCESSED_PATH = os.path.join("preprocessed", "mnist_preprocessed.npz")
    preprocess_data()