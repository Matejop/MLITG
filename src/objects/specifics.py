from math_operations import MathOperations as MO
from typing import List

class ModelSpecifics():         
    def __init__(self, inner_layers: List[int], hyperparameter_variation: int):
        self.inner_layers = [16, 16] if inner_layers == None else inner_layers
        self.inner_layers_size = 0
        self.inner_layer_count = len(self.inner_layers)
        for dimension in self.inner_layers:
            self.inner_layers_size += dimension
        self.inner_layers_size = MO.round(self.inner_layers_size / self.inner_layer_count)
        self.hyperparameters_names = ["batch_size", "gradient_factor", "shift_factor", "bias_spread"]
        self.inner_layer_count += MO.round(hyperparameter_variation / 10)
        for i in range(self.inner_layer_count):
            self.hyperparameters_names.append(f"{i}")