from math_operations import MathOperations as MO
from datetime import datetime as Date
from copy import copy, deepcopy
from session import Session
from typing import List
import random as rnd
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
        self = cls()
        rnd.seed(seed)
        dimensions = deepcopy(dimensions)
        dimensions.append(10)
        dimensions.insert(0, 784)
        self.id = hashlib.md5(f"{"ITG"}+{self.created_at}".encode()).hexdigest()
        self.path = os.path.join(FOLDER_PATH, f"{self.id}.json")
        self.dimensions = dimensions
        self.biases = [None] * (len(self.dimensions) - 1)
        self.weights = [None] * (len(self.dimensions) - 1)
        for i in range(1, len(self.dimensions)):
            self.biases[i - 1] = [rnd.randint(-bias_spread, bias_spread) for _ in range(self.dimensions[i])]
            self.weights[i - 1] = [
                [rnd.random() for _ in range(self.dimensions[i - 1])] for _ in range(self.dimensions[i])
            ]
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
            
    def fit(self, data: tuple) -> dict:
        """
        trains the model and saves it to local
        """
        exception = None
        try:
            #TODO train here
            result = self.__save()
            if not result["status"]:
                raise Exception(result["exception"])
        except Exception as e:
            exception = e
        return {
            "status": True if exception is None else False,
            "model": self,
            "exceptions": exception
        }
    
    def infer(self, activations: List[float], layer_index = 0) -> Session:
        """
        Forward pass
        Function for inference
        activations is the input layer
        Always call without setting layer_index, this is a recursive function
        """
        if layer_index < len(self.weights) - 1:
            layer_activations = [
                MO.sigmoid(x) for x in MO.vector_addition(MO.matrix_x_vector(self.weights[layer_index], activations), self.biases[layer_index])
            ] 
        else:
            return Session.from_inference(
                    MO.softmax(MO.vector_addition(MO.matrix_x_vector(self.weights[layer_index], activations), self.biases[layer_index])),                    
                    self.id
                )
        layer_index += 1
        return self.infer(layer_activations, layer_index)

    #infrastructure

    def __forward(self, activations: List[List[float]], zeds: List[List[float]], layer_index = 0) -> dict:
        """
        Forward pass
        Function for training
        activations (Input) must look like this:
        activations[0] is the input layer
        Always call specifying zeds = [] (new list - avoids list as a parameter compilation complications)
        Always call without setting layer_index, this is arecursive function
        """
        if layer_index < len(self.weights) - 1:
            zeds.append(MO.vector_addition(MO.matrix_x_vector(self.weights[layer_index], activations[layer_index]), self.biases[layer_index]))
            activations.append([MO.sigmoid(x) for x in zeds[layer_index]])
        else:
            zeds.append(MO.vector_addition(MO.matrix_x_vector(self.weights[layer_index], activations[layer_index]), self.biases[layer_index]))
            activations.append(MO.softmax(zeds[layer_index]))
            return {
                "zeds": zeds,
                "activations": activations, 
                "model_id": self.id
            }
        layer_index += 1
        return self.__forward(zeds, activations, layer_index)
        
    def __back_propagation(self, activations, layer_index=None):
        """
        Layer index will be set internally - recursion
        """

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
            #model_dir = dict(orjson.loads(open(path).read()))
            #self.id = model_dir["id"]
            #self.created_at = model_dir["created_at"]
            #self.path = model_dir["path"]
            #self.dimensions = model_dir["dimensions"]
            #self.biases = model_dir["biases"]
            #self.weights = model_dir["weights"]
        except Exception as e:
            exception = e
            self = Model()
        return {
            "status": True if exception is None else False,
            "exception": exception
        }   
