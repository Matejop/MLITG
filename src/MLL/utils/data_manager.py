from typing import List, Tuple
from MLL.utils.input_validation import InputValidation
import math

class DataManager():
    """
    Utility class for handling dataset transformations.

    Currently provides functionality for batching input data into
    smaller chunks used during training.

    Methods:
        batch_data: Splits dataset into mini-batches.
    """
    def batch_data(x: List[List[float]], y: List[List[int]], batch_size: int) -> List[List[Tuple[List[float], List[int]]]]:
        """
        Splits input and label data into batches.

        Each batch is a list of tuples, where each tuple contains:
            (input_sample, label)

        Args:
            x (List[List[float]]): Input feature dataset.
            y (List[List[int]]): Corresponding labels (typically one-hot encoded).
            batch_size (int): Number of samples per batch.

        Returns:
            List[List[Tuple[List[float], List[int]]]]:
            A list of batches. Each batch is a list of (x, y) tuples.

        Raises:
            Exception:
                - If input and label lengths do not match.
                - If batch_size is not a valid positive integer.
                - If batching fails for any reason.

        Notes:
            - The final batch may be smaller than `batch_size` if the dataset
            size is not perfectly divisible.
            - Guarantees at least one batch is returned, even for small datasets.
        """
        try:
            error_message = InputValidation.check_data_len(x, y)
            if error_message != "":
                raise Exception(error_message)
            error_message = InputValidation.check_boundary_int(batch_size, "Batch size", 0)
            if error_message != "":
                raise Exception(error_message)
            batch_count = math.ceil(len(x) / batch_size) 
            if batch_count == 0:
                batch_count = 1
            batches = []
            for i in range(batch_count):
                slice_start = i * batch_size
                slice_end = (i + 1) * batch_size
                if slice_end > len(x):
                    slice_end = len(x)
                batch = []
                for j in range(slice_start, slice_end):
                    batch.append((x[j], y[j]))
                batches.append(batch)
            return batches
        except Exception as e:
            raise Exception(f"An exception occured while batching data: Exception: {e}")