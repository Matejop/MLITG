from typing import List
import random as rnd

class Weights:
    def create(dimensions: List[List[int]], seed: str = "0", zeroes: bool = True) -> List[List[List[float]]]:
        rnd.seed(seed)
        values = [None] * (len(dimensions) - 1)
        if zeroes:
            for i in range(1, len(dimensions)):
                values[i - 1] = [
                    [0] for _ in range(dimensions[i])
                ]
        else:
            for i in range(1, len(dimensions)):
                values[i - 1] = [
                    [rnd.random() for _ in range(dimensions[i - 1])] for _ in range(dimensions[i])
                ]
        return values