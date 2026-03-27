from math_operations import MathOperations as MO
from objects.parameters import Parameters
from objects.specifics import ModelSpecifics
from data_manager import DataManager
from datetime import datetime as Date
from typing import List, Tuple
from copy import deepcopy
import hashlib
import orjson
import os

FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../models")

class Model:    
    #TODO add important parameters/constants to global config
    def __init__(self, batch_size: int, gradient_factor: float, shift_factor: float, bias_spread: int,
        hyperparameter_variation: int, variation_factor: float):
        time_now = Date.now().__str__()
        self.id = hashlib.md5(f"{"ITG"}+{time_now}".encode()).hexdigest()
        self.created_at = time_now
        self.dimensions = None
        self.hyperparameter_variation = hyperparameter_variation 
        self.variation_factor = variation_factor
        self.batch_size = batch_size
        self.gradient_factor = gradient_factor        
        self.shift_factor = shift_factor
        self.bias_spread = bias_spread
        self.parameters = None

    def __set(self, input_dimension: int, output_dimension: int, inner_layers: List[int], seed: str):
        dimensions = deepcopy(inner_layers)
        dimensions.insert(0, input_dimension)
        dimensions.append(output_dimension)
        self.dimensions = dimensions
        self.parameters = Parameters.from_dimensions(dimensions, seed, self.bias_spread)
        
    def fit(data: tuple, input_dimension: int, output_dimension: int, seed: str = "0",
        batch_size: int = 10, gradient_factor: float = 0.9, 
        bias_spread: int = 10, inner_layers: List[int] = None, shift_factor: float = 0.9,
        hyperparameter_variation: int = 100, variation_factor: float = 1) -> tuple:
        """
        trains the model and saves it to local
        uses grid search validation
        """                
        #add wrong input handling
        print("preparing")
        training_data, validation_data, testing_data = data
        base_model = Model(batch_size, gradient_factor, shift_factor, bias_spread, hyperparameter_variation, variation_factor)
        specifics = ModelSpecifics(inner_layers, hyperparameter_variation)
        best_success = 0
        best_model = deepcopy(base_model)
        print("running")
        for i in range(len(specifics.hyperparameters_names) - 1, -1, -1):
            for j in range(len(specifics.hyperparameters_names) - i):
                combination = specifics.hyperparameters_names[:i]
                combination.append(specifics.hyperparameters_names[i + j])
                if i < 4:
                    inner_layers = [specifics.inner_layers_size for _ in range(specifics.inner_layer_count)]
                else:
                    inner_layers = [specifics.inner_layers_size for _ in range(specifics.inner_layer_count - int(specifics.hyperparameters_names[i + j]))]
                model = deepcopy(base_model)
                set = False
                for _ in range(hyperparameter_variation):
                    if not set:
                        model.__set(input_dimension, output_dimension, inner_layers, seed)
                        set = True
                    model.__train(training_data)
                    success = 0
                    for datum in validation_data:
                        classification = model.infer(datum[0])
                        if classification.index(MO.max(classification)) == datum[1]:
                            success += 1
                    success /= len(validation_data)
                    if success > best_success:                    
                        best_success = success
                        best_model = deepcopy(model)
                        print("YAY")
                        print(f"Best success:{best_success}")                    
                        print(f"Hyperparameters:{combination}")
                        print("--------------------------------------------------------")
                    else:
                        print("NAY")
                        print(f"success:{success}")                    
                        print(f"Hyperparameters:{combination}")
                        print("--------------------------------------------------------")
                    #shifting hyperparameters
                    for hyperparameter in combination:
                        if hyperparameter == "batch_size":
                            model.batch_size += MO.round(1 * variation_factor)
                        elif hyperparameter == "gradient_factor":
                            model.gradient_factor += 0.1 * variation_factor
                        elif hyperparameter == "shift_factor":   
                            model.shift_factor += 0.1 * variation_factor
                        elif hyperparameter == "bias_spread":
                            model.bias_spread += MO.round(1 * variation_factor)
                        else:
                            index = int(hyperparameter)
                            inner_layers[index] += MO.round(1 * variation_factor)
                            set = False
        testing_result = best_model.__test(testing_data)
        save_result = best_model.__save()
        if save_result["status"] == False:
            print(save_result["exception"])
            print("Model was not saved succesfully")
        return (best_model, testing_result)
    
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
        training_data = DataManager.make_batches(training_data, self.batch_size)  
        for batch in training_data:      
            for i in range(len(batch)):
                activations = self.__forward([batch[i][0]])
                loss = MO.cross_entropy(activations[-1], batch[i][1]) #applying loss function to the last layer
                loss[batch[i][1]] = loss[batch[i][1]] - 1 #delta for the last last layer
                self.__back_propagation(activations, loss)
            for i in range(len((self.dimensions)) - 1): #calculating the average shift and factoring in shift factor to change learning rate
                self.parameters.biases[i] = MO.vector_addition(self.parameters.biases[i], MO.vector_scalar_product(self.parameters.biases[i], (self.gradient_factor / self.batch_size) * self.shift_factor))
                for j in range(len(self.parameters.weights[i])):
                    self.parameters.weights[i][j] = MO.vector_addition(self.parameters.weights[i][j], MO.vector_scalar_product(self.parameters.weights[i][j], (self.gradient_factor / self.batch_size) * self.shift_factor))
            self.shift_factor *= self.shift_factor
    
    def __test(self, testing_data: List[Tuple[List[float], int]]) -> Tuple[List[List[float]], float]:
        classifications = []
        success = 0
        for datum in testing_data:
            classification = self.infer(datum[0])
            classifications.append(classification)
            if classification.index(MO.max(classification)) == datum[1]:
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
        if layer_index < len(self.parameters.weights) - 1:
            activations.append([MO.sigmoid(x) for x in MO.vector_addition(MO.vector_matrix_product(activations[layer_index], self.parameters.weights[layer_index]), self.parameters.biases[layer_index])])
        else:
            activations.append(MO.softmax(MO.vector_addition(MO.vector_matrix_product(activations[layer_index], self.parameters.weights[layer_index]), self.parameters.biases[layer_index])))
            return activations
        layer_index += 1
        return self.__forward(activations, layer_index)
        
    def __back_propagation(self, activations: List[List[float]], delta: List[float], layer_index: int = 1):
        """
        Back propagation
        Function for training
        activations must be from __forward
        Always call without setting layer_index, this is arecursive function
        Always call without setting loss, this is a recursive function
        """
        #delta is desired change for single layers (partial derivation of Z() to loss)

        #updating biases and weights with delta
        self.parameters.biases[-layer_index] = MO.vector_addition(self.parameters.biases[-layer_index], delta)
        self.parameters.weights[-layer_index] = MO.update_matrix(
            activations[-(layer_index + 1)], 
            MO.update_matrix(
                delta, 
                self.parameters.weights[-layer_index],
                self.parameters.transposed_map[-layer_index]
            )
        )
        if len(activations) - (layer_index + 1) == 0:
            return
        #partial derivation of Z to loss:
        #product of weights and delta from last layer
        delta = MO.vector_matrix_product(
            delta, 
            self.parameters.weights[-layer_index],
            self.parameters.transposed_map[-layer_index]
        )
        #partial derivation of Z to loss - continuing:
        #product of components - hadamard product
        #product of derivation of sigmoid with every component of delta
        delta = [
            delta[i] * 
            (activations[-(layer_index + 1)][i] * (1 - activations[-(layer_index + 1)][i])) #derivation of sigmoid
            for i in range(len(activations[-(layer_index + 1)]))
        ]
        
        layer_index += 1
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