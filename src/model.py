from infrastructure.model_shaper import ModelShaper as MO
import numpy as np

class Model:
    def forward(activations: np.ndarray, weights: np.ndarray, biases: np.ndarray, layer_index = 0) -> np.ndarray:
        """
        Forward pass
        Function for inference
        Always call without setting layer_index - recursive function
        The model has to be formatted to matrices using the ModelShaper.to_matrices() function
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
        The model has to be formatted to matrices using the ModelShaper.to_matrices() function
        Activations is the input
        Activations (Input) must look like this:
        Activations = np.empty(len(weights))
        Activations[0] = input_layer
        """
        activations[layer_index] = np.array
        (
            [
                MO.sigmoid(x) for x in MO.vector_addition(MO.matrix_x_vector(weights[layer_index], activations[layer_index - 1]), biases[layer_index])
            ]
        )
        layer_index += 1
        if layer_index < len(weights):
            return Model.forward(activations[layer_index], weights, biases, layer_index)
        else:
            return activations
        
    def back_probagation(activations, weights: np.ndarray, biases: np.ndarray, layer_index=None):
        """
        Layer index will be set internally - recursion
        """
        