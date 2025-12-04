from infrastructure.math_operations import MathOperations as MO
from infrastructure.file_manager import FileManager as FM
from datetime import datetime as Date
import random as rnd
import numpy as np
import hashlib

class Model: 
    #TODO update docstrings when model_shaper.py todos are done
    def forward(activations: np.ndarray, weights: np.ndarray, biases: np.ndarray, layer_index = 0) -> np.ndarray:
        """
        Forward pass
        Function for inference
        Always call without setting layer_index - recursive function
        Activations is the input layer
        """
        activations = np.array
        (
            [
                MO.sigmoid(x) for x in MO.vector_addition(MO.matrix_x_vector(weights[layer_index], activations), biases[layer_index])
            ]
        )
        layer_index += 1
        if layer_index < len(weights):
            return Model.forward(activations, weights, biases, layer_index)
        else:
            return activations
        
    def forward_train(activations: np.ndarray, weights: np.ndarray, biases: np.ndarray, layer_index = 1) -> np.ndarray:
        """
        Forward pass
        Function for training
        Always call without setting layer_index - recursive function
        activations (Input) must look like this:
        activations = np.empty(len(weights))
        activations[0] = input_layer
        """
        activations[layer_index] = np.array
        (
            [
                MO.sigmoid(x) for x in MO.vector_addition(MO.matrix_x_vector(weights[layer_index], activations[layer_index - 1]), biases[layer_index])
            ]
        )
        layer_index += 1
        if layer_index < len(weights):
            return Model.forward_train(activations[layer_index], weights, biases, layer_index)
        else:
            return activations
        
    def back_probagation(activations, weights: np.ndarray, biases: np.ndarray, layer_index=None):
        """
        Layer index will be set internally - recursion
        """
    def fit(model: dict, data: tuple) -> dict:
        """
        Docstring for fit
        
        :param model: Description
        :type model: dict
        :return: Description
        :rtype: dict
        """

    def create(name = "", dimensions = [16, 16], bias_spread = 10) -> dict:
        #TODO train the model during creation
        """
        dimensions are just hidden layers of the neural network
        """
        model = {}
        exception = None
        try:
            dimensions.append(10)
            dimensions.insert(0, 784)
            biases = np.empty(len(dimensions) - 1, dtype=np.ndarray)
            weights = np.empty(len(dimensions) - 1, dtype=np.ndarray)
            for i in range(1, len(dimensions)):
                biases[i - 1] = [rnd.randint(-bias_spread, bias_spread) for _ in range(dimensions[i])]
                weights[i - 1] = np.array(
                    [[rnd.random() for _ in range(dimensions[i - 1])] for _ in range(1, dimensions[i])]
                )
            time_now = Date.now()
            model = {
                "name": "".join([c if (ord(c) > 96 and ord(c) < 123) or (ord(c) > 64 and ord(c) < 91) else "#" for c in name]), #c stands for char,
                "id": hashlib.md5(f"{name}+{time_now}".encode()).hexdigest(),
                "created_at": "".join(["-" if c == ":" else c for c in str(time_now)]), #c stands for char
                "path": "",
                "dimensions": dimensions,
                "biases": biases,
                "weights": weights
            }
            result = FM.save_model(model)
            if not result["status"]:
                raise Exception(result["exception"])
            model = result["model"]
        except Exception as e:
            exception = e
        return {
            "status": True if exception is None else False,
            "model": model,
            "exceptions": exception
        }