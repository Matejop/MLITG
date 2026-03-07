"""
This file is a library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

import numpy as np

class Loader:
    #TODO add filepath to global config
    def load_data(filepath="./data/preprocessed/mnist_preprocessed.npz"):
        """Return a tuple containing ``(training_data, validation_data,
        test_data)``. Based on ``load_data``, but the format is more
        convenient for use in our implementation of neural networks.

        In particular, ``training_data`` is a list containing 50,000
        2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
        containing the input image.  ``y`` is a 10-dimensional
        numpy.ndarray representing the unit vector corresponding to the
        correct digit for ``x``.

        ``validation_data`` and ``test_data`` are lists containing 10,000
        2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
        numpy.ndarry containing the input image, and ``y`` is the
        corresponding classification, i.e., the digit values (integers)
        corresponding to ``x``.

        Obviously, this means we're using slightly different formats for
        the training data and the validation / test data.  These formats
        turn out to be the most convenient for use in our neural network
        code."""
        data = np.load(filepath)
        training_data = list(zip(
            [x.reshape((784,1)) for x in data['train_x']],
            [y.reshape((10,1)) for y in data['train_y']]
        ))
        validation_data = list(zip(
            [x.reshape((784,1)) for x in data['val_x']],
            data['val_y']
        ))
        test_data = list(zip(
            [x.reshape((784,1)) for x in data['test_x']],
            data['test_y']
        ))
        return (training_data, validation_data, test_data)
