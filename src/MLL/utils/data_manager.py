from typing import List, Tuple
from input_validation import InputValidation
import math

class DataManager():
    def batch_data(x: List[List[float]], y: List[List[int]], batch_size: int) -> List[List[Tuple[List[float], List[int]]]]:
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