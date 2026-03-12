"""
This file is a library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

from typing import Tuple, List
import orjson
import gzip
import os

DATA_PATH = os.path.join("data", "mnist_preprocessed.gz")

class Manager:
    #TODO add filepath to global config
    def load_data(filepath=DATA_PATH) -> Tuple[List[Tuple[List[float], int]]]:
        data = orjson.loads(gzip.open(filepath).read())
        training_data = list(zip(
            data["training_x"],
            data["training_y"]
        ))
        validation_data = list(zip(
            data['validation_x'],
            data['validation_y']
        ))
        test_data = list(zip(
            data['testing_x'],
            data['testing_y']
        ))
        return (training_data, validation_data, test_data)
    
    def make_batches(training_data: List[Tuple[List[float], int]], batch_size: int) -> List[List[Tuple[List[float], int]]]:
        length_ratio = 0
        if batch_size < 1:
            print("Batch size incorrectly defined - batches not created")
            print(batch_size)
            return []
        while length_ratio < len(training_data) / batch_size:
            length_ratio += 1
        length_ratio -= 1
        batched_data = []
        for i in range(length_ratio):
            slice_start = i * batch_size
            slice_end = (i + 1) * batch_size
            if slice_end > len(training_data):
                slice_end = len(training_data)
            batched_data.append((training_data[slice_start:slice_end]))
        return batched_data
                