from math_operations import MathOperations as MO
from objects.parameters import Parameters
from objects.hyperparameters import Hyperparameters
from data_manager import Manager
from datetime import datetime as Date
from copy import deepcopy
from typing import List, Tuple
import hashlib
import orjson
import os

FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../models")

class Model:    
    #TODO add important parameters/constants to global config
    def __init__(self):
        self.id = None
        self.created_at = None
        self.dimensions = None
        self.path = None
        self.batch_size = None
        self.gradient_factor = None
        self.shift_factor = None
        self.bias_spread = None
        self.parameters = None
    
    @classmethod
    def from_dimensions(cls, input_dimension, output_dimension, hyperparameters: Hyperparameters, seed: str = "0"):
        self = cls()
        dimensions = hyperparameters.inner_layers        
        dimensions.insert(0, input_dimension)
        dimensions.append(output_dimension)
        
        time_now = Date.now().__str__()
        self.id = hashlib.md5(f"{"ITG"}+{time_now}".encode()).hexdigest()
        self.created_at = time_now
        self.dimensions = dimensions
        self.path = os.path.join(FOLDER_PATH, f"{self.id}.json")
        self.batch_size = hyperparameters.batch_size
        self.gradient_factor = hyperparameters.gradient_factor
        self.shift_factor = hyperparameters.shift_factor
        self.bias_spread = hyperparameters.bias_spread
        self.parameters = Parameters.from_dimensions(self.dimensions, seed, self.bias_spread)
        return self

    def fit(data: tuple, input_dimension: int, output_dimension: int, seed: str = "0",
        batch_size: int = 100, gradient_factor: float = 1, shift_factor: float = 0.9,
        bias_spread: int = 10, inner_layers: List[int] = None,
        hyperparameter_variation: int = 100, variation_factor: float = 1) -> tuple:
        """
        trains the model and saves it to local
        uses grid search validation
        """                
        #add wrong input handling
        training_data, validation_data, testing_data = data
        combinations = Hyperparameters.make_combinations(hyperparameter_variation, variation_factor, batch_size, gradient_factor, shift_factor, bias_spread, inner_layers)  
        best_success = 0
        fitted_model = None
        for i in range(len(combinations)):
            for _ in range(hyperparameter_variation):
                model = Model.from_dimensions(input_dimension, output_dimension, combinations[i], seed).__train(training_data)
                classification_success = 0
                for datum in validation_data:
                    classification_success += model.infer(datum[0])[datum[1]]
                classification_success /= len(validation_data)
                if classification_success > best_success:
                    best_success = classification_success
                    fitted_model = model
                combinations[i].shift_values()
        testing_result = fitted_model.__test(testing_data)
        save_result = fitted_model.__save()
        if save_result["status"] == False:
            print(save_result["exception"])
            print("Model was not saved succesfully")
        return (fitted_model, testing_result)
    
    def infer(self, activations: List[float], layer_index: int = 0) -> List[float]:
        """
        Function for inference
        activations is the input layer
        Always call without setting layer_index, this is a recursive function
        """
        if layer_index < len(self.parameters.weights) - 1:
            layer_activations = [
                MO.sigmoid(x) for x in MO.vector_addition(MO.vector_matrix_product(activations, self.parameters.weights[layer_index]), self.parameters.biases[layer_index])
            ] 
        else:
            return MO.softmax(MO.vector_addition(MO.vector_matrix_product(activations, self.parameters.weights[layer_index]), self.parameters.biases[layer_index]))
        layer_index += 1
        return self.infer(layer_activations, layer_index)

    #infrastructure

    def __train(self, training_data: List[Tuple[List[float], int]]):
        training_data = Manager.make_batches(training_data, self.batch_size)  
        for batch in training_data:      
            for i in range(len(batch)):
                activations = self.__forward([batch[i][0]])
                loss = MO.cross_entropy(activations.pop(), batch[i][1]) #applying loss function to the last layer
                loss[batch[i][1]] = loss[batch[i][1]] - 1 #delta for the last last layer
                self.__back_propagation(activations, loss)
            for i in range(len((self.dimensions))): #calculating the average shift and factoring in shift factor to change learning rate
                self.parameters.biases[i] = MO.vector_addition(self.parameters.biases[i], MO.vector_scalar_product(self.parameters.biases[i], (self.gradient_factor / self.batch_size) * self.shift_factor))
                for k in range(len(self.parameters.weights[i])):
                    self.parameters.weights[i][k] = MO.vector_addition(self.parameters.weights[i][k], MO.vector_scalar_product(self.parameters.weights[i][k], (self.gradient_factor / self.batch_size) * self.shift_factor))
            self.shift_factor *= self.shift_factor
        return self
    
    def __test(self, testing_data: List[Tuple[List[float], int]]) -> Tuple[List[List[float]], float]:
        classifications = []
        success = 0
        for datum in testing_data:
            classification = self.infer(datum[0])
            classifications.append(classification)
            if MO.max(classification[datum[1]]) == classification[datum[1]]:
                success += 1
        return (classifications, success / len(testing_data))

    def __forward(self, activations: List[List[float]], layer_index: int = 0) -> List[List[float]]:
        """
        Forward pass
        Function for training
        activations (Input) must look like this:
        activations[0] is the input layer
        Always call without setting layer_index, this is a recursive function
        """    
        if layer_index + 1 < len(self.parameters.weights) - 1:
            if layer_index == 0:
                activations.append([MO.sigmoid(x) for x in MO.vector_addition(MO.vector_matrix_product(activations.pop(), self.parameters.weights[layer_index]), self.parameters.biases[layer_index])])
                layer_index -= 1
                return self.__forward(activations, layer_index)
            elif layer_index == -1:
                layer_index = 0
            activations.append([MO.sigmoid(x) for x in MO.vector_addition(MO.vector_matrix_product(activations[layer_index], self.parameters.weights[layer_index + 1]), self.parameters.biases[layer_index + 1])])
        else:
            activations.append(MO.softmax(MO.vector_addition(MO.vector_matrix_product(activations[layer_index], self.parameters.weights[layer_index + 1]), self.parameters.biases[layer_index + 1])))
            return activations
        layer_index += 1
        return self.__forward(activations, layer_index)
        
    def __back_propagation(self, activations: List[List[float]], delta: List[float], layer_index: int = -1):
        """
        Back propagation
        Function for training
        activations must be from __forward
        Always call without setting layer_index, this is arecursive function
        Always call without setting loss, this is arecursive function
        """


        #index problems here


        if layer_index == -1:
            layer_index = len(activations) - 1

        #delta is desired change for single layers (partial derivation of Z() to loss)
        #partial derivation of Z to loss:
        #product of weights (this layer) and delta from last layer
        delta = MO.vector_matrix_product(
            delta, 
            self.parameters.weights[layer_index + 1],
            self.parameters.transposed_map[layer_index + 1]
        )
        #partial derivation of Z to loss - continuing:
        #product of components - hadamard product
        #product of derivation of sigmoid with every component of delta
        delta = [
            delta[i] * 
            (activations[layer_index][i] * (1 - activations[layer_index][i])) #derivation of sigmoid
            for i in range(len(activations[layer_index]))
        ]
        #updating biases and weights with delta
        self.parameters.biases[layer_index + 1] = MO.vector_addition(self.parameters.biases[layer_index + 1], delta)
        self.parameters.weights[layer_index + 1] = MO.update_matrix(
            activations[layer_index], 
            MO.update_matrix(delta, self.parameters.weights[layer_index + 1]),
            self.parameters.transposed_map[layer_index + 1]
        )

        if layer_index == 0:
            return
        
        layer_index -= 1
        return self.__back_propagation(activations, delta, layer_index)

    def __save(self) -> dict:
        exception = None
        model_dict = self.__dict__        
        model_dict["biases"] = self.parameters.biases
        model_dict["weights"] = self.parameters.weights
        model_dict.__delitem__("parameters")
        try:
            if os.path.exists(self.path) == False:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as f:
                f.write(orjson.dumps(
                    model_dict, 
                    option=orjson.OPT_INDENT_2).decode()
                )
        except Exception as e:
            exception = e
        return {
            "status": True if exception is None else False,
            "exception": exception
        }
        
    def __load(self, path: str) -> dict:
        exception = None
        try:
            model_dict = dict(orjson.loads(open(path).read()))
            model_dict["parameters"] = Parameters(model_dict["biases"], model_dict["weights"])
            model_dict.__delitem__("biases")
            model_dict.__delitem__("weights")
            self.__dict__ = model_dict
        except Exception as e:
            exception = e
            self = Model()
        return {
            "status": True if exception is None else False,
            "exception": exception
        }   