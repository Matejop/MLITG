from typing import List
import random as rnd

class Weights:
    def __init__(self, values: List[List[float]]):
        self.values = values
        self.transposed_map = [[[(k, j) for k in range(len(values[i]))] for j in range(len(values[i][0]))] for i in range(len(values))] 

    @classmethod
    def from_dimensions(cls, dimensions: List[List[int]], seed: str = "0", zeroes: bool = True):
        rnd.seed(seed)
        self = cls([[[0 for _ in range(dimensions[i - 1])] for j in range(dimensions[i])] for i in range(1, len(dimensions))])
        if not zeroes: 
            for i in range(len(self.values)):
                for j in range(len(self.values[i])):
                    for k in range(len(self.values[i][j])):
                        self.values[i][j][k] = rnd.random()
        return self