from math_operations import MathOperations as MO
from typing import List, Dict

class Hyperparameters():
    def __init__(self, keys: List[str], hashmap: Dict[str, float]):
        self.keys = keys
        self.hashmap = hashmap

    def from_input(batch_size, gradient_factor, bias_spread, inner_layer_count, inner_layers_size):
        keys = ["batch_size", "gradient_factor", "bias_spread"]
        hashmap = {
            "batch_size": batch_size,
            "gradient_factor": gradient_factor,
            "bias_spread": bias_spread,
        }
        #Adding each inner layer as a seperate hyperparameter
        for i in range(inner_layer_count):
            keys.append(f"{i}")
            hashmap[f"{i}"] = inner_layers_size
        return Hyperparameters(keys, hashmap)
    
    def extract_combination(self, i: int, j: int):
        keys = []
        hashmap = dict() 
        for k in range(i):
            keys.append(self.keys[k])
            hashmap[self.keys[k]] = self.hashmap[self.keys[k]]
        keys.append(self.keys[i + j])
        hashmap[self.keys[i + j]] =  self.hashmap[self.keys[i + j]]
        return Hyperparameters(keys, hashmap)