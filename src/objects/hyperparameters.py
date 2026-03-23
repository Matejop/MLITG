from math_operations import MathOperations as MO
from typing import List

class Hyperparameters():    
    def __init__(self, variation_factor: float, batch_size: int, gradient_factor: float, 
            shift_factor: float, bias_spread: int, inner_layers: List[int], combination: List[str]):
        inner_layer_count = 0
        for hyperparameter in combination:
            if hyperparameter[:11] == "inner_layer":
                index = int(hyperparameter.split(":")[1])
                if inner_layer_count < index:
                    inner_layer_count = index + 1
        self.variation_factor = variation_factor 
        self.batch_size = batch_size
        self.gradient_factor = gradient_factor
        self.shift_factor = shift_factor
        self.bias_spread = bias_spread
        self.inner_layers = inner_layers[:inner_layer_count]
        self.combination = combination

    @classmethod
    def make_combinations(cls, hyperparameter_variation: int, variation_factor: float, 
        batch_size: int, gradient_factor: float, shift_factor: float, 
        bias_spread: int, inner_layers: List[int]) -> list:

        inner_layers = inner_layers if inner_layers != None else [16, 16]
        inner_layers_count = len(inner_layers)
        inner_layers_average_size = 0
        for dimension in inner_layers:
            inner_layers_average_size += dimension
        inner_layers_average_size = MO.round(inner_layers_average_size / inner_layers_count)

        combinations = []
        hyperparameters_names = ["batch_size", "gradient_factor", "shift_factor", "bias_spread"]
        for i in range(inner_layers_count + MO.round(hyperparameter_variation / 10)):
            hyperparameters_names.append(f"inner_layer:{i}")
            if i > inner_layers_count - 1:
                inner_layers.append(inner_layers_average_size)
        for i in range(len(hyperparameters_names) - 1, -1, -1):
            for j in range(len(hyperparameters_names) - i):
                combination = hyperparameters_names[:i]
                combination.append(hyperparameters_names[i + j])
                combinations.append(cls(variation_factor, batch_size, gradient_factor, shift_factor, bias_spread, inner_layers, combination)) 
        return combinations
    
    def shift_values(self):
        found = []
        shifted = False
        for hyperparameter in self.combination:
            for x in found:
                if hyperparameter == x:
                    shifted == True
                    break
            if shifted == False:
                if hyperparameter == "batch_size":
                    self.batch_size += MO.round(1 * self.variation_factor)
                elif hyperparameter == "gradient_factor":
                    self.gradient_factor += 0.1 * self.variation_factor
                elif hyperparameter == "shift_factor":
                    self.shift_factor += 0.1 * self.variation_factor           
                elif hyperparameter == "bias_spread":
                    self.bias_spread += MO.round(1 * self.variation_factor)
                elif hyperparameter[:11] == "inner_layer":
                    index = hyperparameter.split(":")[1]
                    self.inner_layers[index] += MO.round(1 * self.variation_factor)
                found.append(hyperparameter)