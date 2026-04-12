from typing import List
import math
import random as rnd

class Parameters:
    def __init__(self, biases: List[List[float]], weights: List[List[List[float]]]):
        self.biases = biases
        self.weights = weights

    def from_dimensions(dimensions: List[List[int]], zeroes: bool, bias_spread: int = 10, seed: str = "0"):
        rnd.seed(seed)
        if not zeroes:
            biases = []
            weights = []
            for i in range(1, len(dimensions)):
                biases.append([])
                weights.append([])
                for j in range(dimensions[i - 1]):                                     
                    weights[i - 1].append([])
                    for _ in range(dimensions[i]):
                        limit = math.sqrt(6 / (dimensions[i - 1] + dimensions[i]))                                          
                        weights[i - 1][j].append(rnd.uniform(-limit, limit))
                for j in range(dimensions[i]):
                    biases[i - 1].append(rnd.uniform(-bias_spread, bias_spread))
            parameters = Parameters(
                biases,
                weights
            )
            return parameters
        else:
            parameters = Parameters(
                [[0.0 for _ in range(dimensions[i])] for i in range(1, len(dimensions))],
                [[[0.0 for _ in range(dimensions[i])] for _ in range(dimensions[i - 1])] for i in range(1, len(dimensions))]
            )
            return parameters
    def trial():#
        biases = [[4, 5], [7, -3]]#
        weights = [[[0.2, 0.1], [0.4, 0.3], [0.8, 0.5], [0.6, 0.2]], [[0.6, 0.4], [0.4, 0.2]]]#
        return Parameters(#
            biases,#
            weights#
        )#
    def super_trial():#
        biases = [[4, 2]]#
        weights = [[[4, 2], [5, 1], [8, 7], [0, 6]]]#
        parameters = Parameters(#
            biases,#
            weights#
        )#
        return parameters#
    
if __name__ == "__main__":#
    params = Parameters.from_dimensions([4, 2, 2], False)#
 