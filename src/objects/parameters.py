from typing import List
import random as rnd

class Parameters:
    def __init__(self, biases: List[List[float]], weights: List[List[List[float]]]):
        self.biases = biases
        self.weights = weights
        self.transposed_map = [[[(k, j) for k in range(len(weights[i]))] for j in range(len(weights[i][0]))] for i in range(len(weights))]
    def from_dimensions(dimensions: List[List[int]], seed: str, bias_spread: int):
        rnd.seed(seed)
        self = Parameters(
            [[rnd.randint(-bias_spread, bias_spread) for _ in range(dimensions[i])] for i in range(1, len(dimensions))],
            [[[rnd.random() for _ in range(dimensions[i - 1])] for _ in range(dimensions[i])] for i in range(1, len(dimensions))]
        )
        return self