from typing import List
import random as rnd

class Biases:
    def __init__(self, values: List[List[float]]):
        self.values = values

    @classmethod
    def from_dimensions(cls, dimensions: List[List[int]], seed: str = "0", bias_spread: int = -1) -> List[List[float]]:
        rnd.seed(seed)
        self = cls([None] * (len(dimensions) - 1))
        if bias_spread == -1:
            for i in range(1, len(dimensions)):
                self.values[i - 1] = [0.0 for _ in range(dimensions[i])]
        else:
            for i in range(1, len(dimensions)):
                self.values[i - 1] = [rnd.randint(-bias_spread, bias_spread) for _ in range(dimensions[i])]
        return self
