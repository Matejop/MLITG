from math_operations import MathOperations as MO
from paramaters.biases import Biases
from paramaters.weights import Weights
from data_manager import Manager
from datetime import datetime as Date
from copy import copy, deepcopy
from typing import List, Tuple
import hashlib
import orjson
import os

FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../models")

class Model:    
    #TODO add important parameters/constants to global config
    #Model initialization, infrastructure and object handling will be moved to a different branch once a model is trained

    models = {}

    def __init__(self):
        self.id = None
        self.created_at = self.created_at = Date.now().__str__()
        self.path = None
        self.dimensions = None
        self.biases = None
        self.weights = None
    
    @classmethod
    def from_dimensions(cls, dimensions: List[float], bias_spread = 10, seed = "0"):
        dimensions = deepcopy(dimensions)
        for i in range(len(dimensions)):
            if dimensions[i] < 1:
                print("Dimensions incorrectly defined - model not created")
                print(dimensions)
                return cls()
        self = cls()
        self.id = hashlib.md5(f"{"ITG"}+{self.created_at}".encode()).hexdigest()
        self.path = os.path.join(FOLDER_PATH, f"{self.id}.json")
        self.dimensions = dimensions
        self.biases = Biases.create(self.dimensions, seed, bias_spread)
        self.weights = Weights.create(self.dimensions, seed, False)
        result = self.__save()
        if result["status"]:
            Model.models[self.id] = copy(self)
            return self
        else:
            print(result["exception"])
            return cls()
    
    @classmethod
    def from_path(cls, path: str):
        self = cls()
        result = self.__load(path)
        if result["status"]:
            Model.models[self.id] = copy(self)
        else:
            print(result["exception"])            
        return self
    
    @classmethod
    def from_id(cls, model_id: str):
        return Model.models.get(model_id, cls())
            
    def fit(self, data: tuple, batch_size: int = 100) -> dict:
        """
        trains the model and saves it to local
        """
        training_data, validation_data, testing_data = data
        training_data = Manager.make_batches(training_data, batch_size)
        biases_stencil = Biases.create(self.dimensions)
        weights_stencil = Weights.create(self.dimensions)
        for batch in training_data:
            biases_averages = biases_stencil
            weights_averages = weights_stencil            
            for i in range(len(batch)):
                activations = self.__forward(training_data[0][i], [])
                loss = MO.cross_entropy(activations.pop(), batch[i]) #applying loss function to the last layer
                loss[batch[i]] = loss[batch[i]] - 1 #delta for the last last layer
                biases_averages, weights_averages = self.__back_propagation(activations, loss, biases_averages, weights_averages)
            for i in range(len(weights_averages)):
                self.weights[i] = MO.vector_addition(self.weights[i], weights_averages[i])
                self.biases[i] = biases_averages[i]
        #Add optimizing using validation data
        #Compare false positives, true positives, false negatives and true negatives using testing data
        result = self.__save()
        if not result["status"]:
            print(result["exception"])
            print("The file where the model is saved and its parameters may have been corrupted")
        return self
    
    def infer(self, activations: List[float], layer_index: int = 0) -> List[float]:
        """
        Function for inference
        activations is the input layer
        Always call without setting layer_index, this is a recursive function
        """
        if layer_index < len(self.weights) - 1:
            layer_activations = [
                MO.sigmoid(x) for x in MO.vector_addition(MO.matrix_x_vector(self.weights[layer_index], activations), self.biases[layer_index])
            ] 
        else:
            return MO.softmax(MO.vector_addition(MO.matrix_x_vector(self.weights[layer_index], activations), self.biases[layer_index]))
        layer_index += 1
        return self.infer(layer_activations, layer_index)

    #infrastructure

    def __forward(self, activations: List[List[float]], layer_index: int = 0) -> List[List[float]]:
        """
        Forward pass
        Function for training
        activations (Input) must look like this:
        activations[0] is the input layer
        Always call with setting zeds = [] (avoiding list as a parameter complication)
        Always call without setting layer_index, this is arecursive function
        """    

        if layer_index < len(self.weights) - 1:
            if layer_index == 0:
                activations = activations[1:]
            activations.append([MO.sigmoid(x) for x in MO.vector_addition(MO.matrix_x_vector(self.weights[layer_index], activations[layer_index]), self.biases[layer_index])])
        else:
            activations.append(MO.softmax(MO.vector_addition(MO.matrix_x_vector(self.weights[layer_index], activations[layer_index]), self.biases[layer_index])))
            return activations
        layer_index += 1
        return self.__forward(activations, layer_index)
        
    def __back_propagation(self, activations: List[List[float]], delta: List[float], biases: List[List[int]], weights: List[List[List[float]]], layer_index: int = -1) -> Tuple:
        """
        Back propagationg
        Function for training
        activations must be from __forward
        Always call without setting layer_index, this is arecursive function
        Always call without setting loss, this is arecursive function
        """
        if layer_index == -1:
            layer_index == len(activations) - 1
            delta = []
            layer_index -= 1
        elif layer_index > 0:
            layer_index -= 1
        else:      
            print()

    def __save(self) -> dict:
        exception = None
        try:
            if not os.path.exists(self.path):
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as f:
                f.write(orjson.dumps(
                    self.__dict__, 
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
            self.__dict__ = dict(orjson.loads(open(path).read()))
        except Exception as e:
            exception = e
            self = Model()
        return {
            "status": True if exception is None else False,
            "exception": exception
        }   
