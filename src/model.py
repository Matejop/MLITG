from math_operations import MathOperations as MO
from datetime import datetime as Date
from session import Session
from typing import overload
import random as rnd
import hashlib
import orjson
import os

class Model:    
    #TODO add important parameters/constants to global config
    @overload
    def __init__(self, dimensions = None, bias_spread = 10, seed = "0") -> dict:
        rnd.seed(seed)
        dimensions = dimensions if dimensions != None else [16, 16]
        dimensions.append(10)
        dimensions.insert(0, 784)
        self.created_at = Date.now
        self.id = hashlib.md5(f"{"ITG"}+{self.created_at}".encode()).hexdigest()
        self.path = f"{os.path.join(os.path.dirname(__file__), "..\\models")}\\{self.id}_{self.created_at}.json"
        self.dimensions = dimensions
        self.biases = [None * (len(self.dimensions) - 1)]
        self.weights = [None * (len(self.dimensions) - 1)]
        for i in range(1, len(self.dimensions)):
            self.biases[i - 1] = [rnd.randint(-bias_spread, bias_spread) for _ in range(self.dimensions[i])]
            self.weights[i - 1] = [
                [rnd.random() for _ in range(self.dimensions[i - 1])] for _ in range(1, self.dimensions[i])
            ]
        result = self.__save()
        if not result["status"]:
            self.path = ""
            print(result["exception"])
        

    @overload
    def __init__(self, path: str) -> dict:
        self.id = None
        self.created_at = None
        self.path = None
        self.dimensions = None
        self.biases = None
        self.weights = None
        result = self.__load(path)
        if result["status"]:
            self.id = result["model"]["id"]
            self.created_at = result["model"]["created_at"]
            self.path = result["model"]["path"]
            self.dimensions = result["model"]["dimensions"]
            self.biases = result["model"]["biases"]
            self.weights = result["model"]["weights"]
        else:
            print(result["exception"])
            
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
    
    def forward(self, activations: list, layer_index = 0) -> Session:
        #TODO add softmax to final layer
        """
        Forward pass
        Function for inference
        Always call without setting layer_index - recursive function
        Activations is the input layer
        """
        activations = [
            MO.sigmoid(x) for x in MO.vector_addition(MO.matrix_x_vector(self.weights[layer_index], activations), self.biases[layer_index])
        ]
        layer_index += 1
        if layer_index < len(self.weights):
            return self.forward(activations, layer_index)
        else:
            return Session(activations, self)

    def __forward(self, activations: list, layer_index = 1) -> Session:
        #TODO add softmax to final layer
        #TODO save Z() not just activations
        """
        Forward pass
        Function for training
        Always call without setting layer_index - recursive function
        activations (Input) must look like this:
        activations = [None * len(weights)]
        activations[0] = input_layer
        """
        activations[layer_index] = [
            MO.sigmoid(x) for x in MO.vector_addition(MO.matrix_x_vector(self.weights[layer_index], activations[layer_index - 1]), self.biases[layer_index])
        ]
        layer_index += 1
        if layer_index < len(self.weights):
            return self.__forward(activations, layer_index)
        else:
            return Session(activations, self)
        
    def __back_probagation(self, activations, layer_index=None):
        """
        Layer index will be set internally - recursion
        """

    #infrastructure

    def __save(self) -> dict:
        exception = None
        try:
            if not os.path.exists(self.path):
                os.mkdir(self.path)
            model_dir = {
                "id": self.id,
                "created_at": self.created_at,
                "path": self.path,
                "dimensions": self.dimensions,
                "biases": self.biases,
                "weights": self.weights
            }
            with open(self.path, "w") as f:
                f.write(orjson.dumps(
                    model_dir, 
                    option=orjson.OPT_INDENT_2).decode()
                )
        except Exception as e:
            exception = e
        return {
            "status": True if exception is None else False,
            "exception": exception
        }
        
    def __load(path: str) -> dict:
        exception = None
        try:
            model_dir = dict(orjson.loads(open(path).read()))
        except Exception as e:
            exception = e
        return {
            "status": True if exception is None else False,
            "model_dir": model_dir if exception is None else {},
            "exception": exception
        }   