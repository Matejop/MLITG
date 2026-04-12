from math_operations import MathOperations as MO
from objects.parameters import Parameters
from objects.hyperparameters import Hyperparameters
from data_manager import DataManager
from datetime import datetime as Date
from typing import List, Tuple
from copy import deepcopy
import hashlib
import orjson
import os
import math#

FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../../models")

class Model:    
    #TODO add important parameters/constants to global config
    def __init__(self):
        self.id = None
        self.seed = None
        self.created_at = None
        self.dimensions = None
        self.path = None
        self.hyperparameter_variation = None
        self.variation_factor = None
        self.max_batch_size = None
        self.learning_rate = None        
        self.shift_factor = None
        self.bias_spread = None
        self.parameters = None

    def __from_input(self, seed: str, max_batch_size: int, learning_rate: float, 
        bias_spread: int, hyperparameter_variation: int, 
        variation_factor: float):
        time_now = Date.now().__str__()
        self.id = hashlib.md5(f"{"ITG"}+{time_now}".encode()).hexdigest()
        self.seed = seed
        self.created_at = time_now
        self.dimensions = None
        self.path = os.path.join(FOLDER_PATH, f"{self.id}.json")
        self.hyperparameter_variation = hyperparameter_variation 
        self.variation_factor = variation_factor
        self.max_batch_size = max_batch_size
        self.learning_rate = learning_rate 
        self.bias_spread = bias_spread
        self.parameters = None
        return self

    def __set(self, dimensions: List[int]):
        self.dimensions = dimensions
        self.parameters = Parameters.from_dimensions(dimensions, False, self.bias_spread, self.seed)
        
    def fit(data: tuple, dimensions: List[int] = None, seed: str = "0",
        max_batch_size: int = 500, learning_rate: float = 0.01, bias_spread: int = 0.1,
        hyperparameter_variation: int = 100, variation_factor: float = 0.7):
        """
        trains the model and saves it to local
        uses grid search validation
        """                
        #add wrong input handling
        print("preparing")
        dimensions = [784, 64, 10] if dimensions == None else dimensions
        training_data, validation_data, testing_data = data
        model = Model()
        model = model.__from_input(seed, max_batch_size, learning_rate, bias_spread, hyperparameter_variation, variation_factor)
        model.__set(dimensions)
        print("running")
        model.__train(training_data)
        model.__save()#        
        return model
    
    def infer(self, activations: List[float], layer_index: int = 0) -> List[float]:
        """
        Function for inference
        activations is th input layer
        Always call without setting layer_index, this is a recursive function
        """
        zeds = MO.vector_addition(MO.vector_matrix_product(activations, self.parameters.weights[layer_index]), self.parameters.biases[layer_index])
        if layer_index < len(self.parameters.weights) - 1:
            layer_index += 1
            return self.infer([max(0, x) for x in zeds], layer_index)#maybe change to RElu: MO.sigmoid(x)
        else:
            return MO.softmax(zeds)        

    #infrastructure

    def __train(self, training_data: List[Tuple[List[float], int]]):
        batches = DataManager.make_batches(training_data, self.max_batch_size)  
        counter = 1#
        for batch in batches:  
            loss_avg = 0.0#      
            print(f"Training on batch {len(batch) * (counter - 1)} - {len(batch) * counter}")#
            #step = Parameters.from_dimensions(self.dimensions, True)
            for input in batch:
                activations = []
                activations.append(input[0]) 
                #getting activations by passing forward
                for l in range(len(self.parameters.weights)):
                    zeds = MO.vector_addition(
                        MO.vector_matrix_product(
                            activations[l], 
                            self.parameters.weights[l]), 
                        self.parameters.biases[l]
                    )
                    if l < len(self.parameters.weights) - 1:
                        activations.append([max(0, x) for x in zeds])#maybe change to RElu: #MO.sigmoid(x)
                    else:
                        activations.append(MO.softmax(zeds))
                #backpropagation - gradient descent
                #delta = MO.cross_entropy(activations[-1], input[1])     
                delta = deepcopy(activations[-1])
                delta[input[1]] -= 1
                for l in range(1, len(self.parameters.weights)):  
                    #sum = 0#
                    #for x in delta:#
                    #    sum += abs(x)#
                    #print("delta norm:", sum)#
                    #self.parameters.biases[-l] = MO.vector_addition(self.parameters.biases[-l], delta)
                    for i in range(len(self.parameters.biases)):
                        self.parameters.biases[-l][i] -= delta[i] * self.learning_rate
                    for i, delta_element in enumerate(delta):
                        for j, activation in enumerate(activations[-(l + 1)]):
                            self.parameters.weights[-l][j][i] -= delta_element * activation * self.learning_rate
                    if l + 1 < len(self.parameters.weights):
                        delta = MO.vector_matrix_product(delta, self.parameters.weights[-l], True)
                        delta = [
                            delta[i] 
                            * 0 if max(0, activation) == 0 else 1 #maybe change to RElu: 0 if max(0, activation) == 0 else 1, activation * (1 - activation)
                            for i, activation in enumerate(activations[-(l + 1)])
                        ]
                loss_avg += MO.cross_entropy(activations[-1], input[1])#      
                if type(input[1]) != int:#
                    raise ValueError#                
            #sumx = 0.0#
            #for element in self.weights[0][8]:#
            #    sumx += element#
            #print(f"size of weights layer 1 row 8: {math.sqrt(abs(sumx))}")#
            print(f"loss_avg: {loss_avg / len(batch)}")#
            #print(f"gradient_factor: {gradient_factor}")
            #print("before:")#
            #print(self.parameters.biases[0][8])#
            #print(self.parameters.weights[0][392][8])#
            #changing parameters using values from gradient descent
            #for i in range(len((self.dimensions)) - 1): 
            #    for j in range(len((step.biases[i]))):
            #        #learning_rate = (self.parameters.weights[i][j][k] - (step.weights[i][j][k] / self.batch_size)) * gradient_factor
            #        self.parameters.biases[i][j] -= (step.biases[i][j] / len(batch)) * self.learning_rate
            #    for j in range(len(step.weights[i])):
            #        for k in range(len(step.weights[i][j])):
            #            #learning_rate = (self.parameters.weights[i][j][k] - (step.weights[i][j][k] / self.batch_size) * gradient_factor)
            #            self.parameters.weights[i][j][k] -= (step.weights[i][j][k] / len(batch)) * self.learning_rate
            #print("after:")#    
            #print(self.parameters.biases[0][8])#
            #print(self.parameters.weights[0][392][8])#
            #if last_loss_avg != None:
            #    if (loss_avg / self.batch_size) >= last_loss_avg:
            #       if not shifted:
            #            gradient_factor = self.gradient_factor
            #            shifted = True
            #        else:
            #            gradient_factor *= gradient_factor             
            #last_loss_avg = loss_avg / self.batch_size              
            counter += 1

    #def __forward(self, input_vector: List[float], activations: List[List[float]] = None, layer_index: int = 0) -> List[List[float]]:
    #    """
    #    Forward pass
    #    Function for training
    #    activations (Input) must look like this:
    #    activations[0] is the input layer
    #    Always call without setting layer_index, this is a recursive function
    #    """  
    #    if layer_index < len(self.parameters.weights) - 1:
    #        if layer_index == 0:
    #            activations = []
    #            activations.append(input_vector)
    #        activations.append([MO.sigmoid(x) for x in MO.vector_addition(MO.vector_matrix_product(activations[layer_index], self.parameters.weights[layer_index]), self.parameters.biases[layer_index])])
    #    else:
    #        activations.append(MO.softmax(MO.vector_addition(MO.vector_matrix_product(activations[layer_index], self.parameters.weights[layer_index]), self.parameters.biases[layer_index])))
    #        return activations
    #    layer_index += 1
    #    return self.__forward(input_vector, activations, layer_index)

    #def __back_propagation(self, step: Parameters, activations: List[List[float]], delta: List[float] = None, layer_index: int = None) -> Parameters:
    #    """
    #    Back propagation
    #    Function for training
    #    activations must be from __forward
    #    Always call without setting layer_index, this is arecursive function
    #    Always call without setting loss, this is a recursive function
    #    """
    #    #delta is desired change for single layers (partial derivation of Z() to loss)

        #updating biases and weights with delta  
    #    if layer_index == None:
    #        delta = activations[-1]
    #        delta[batch[i][1]] -= 1   
    #        layer_index = len(self.dimensions) - 2      
    #    step.biases[layer_index] = MO.vector_addition(step.biases[layer_index], delta)
    #    for i in range(len(delta)):
    #        for j in range(len(activations[layer_index - 1])):
    #            step.weights[layer_index][j][i] += delta[i] * activations[layer_index - 1][j]
    #
    #   if layer_index == 1:
    #        return step
        #partial derivation of Z to loss:
        #product of weights and delta from last layer
    #    delta = MO.vector_matrix_product(
    #        delta, 
    #        self.parameters.weights[layer_index],
    #        self.parameters.transposed_map[layer_index]
    #    )
        #delta = MO.vector_matrix_product(
        #    delta,
        #    self.parameters.weights[-layer_index]
        #)
        #partial derivation of Z to loss - continuing:
        #product of components - hadamard product
        #product of derivation of sigmoid with every component of delta
    #    delta = [
    #        delta[i] * 
    #        (activations[layer_index][i] * (1 - activations[layer_index][i])) #derivation of sigmoid
    #        for i in range(len(activations[layer_index]))
    #    ]
    #    layer_index -= 1
    #    return self.__back_propagation(activations, delta, step, layer_index)
    
    def __shift_parameters(self, hyperparameters: List[str]) -> dict:
        holder = deepcopy(hyperparameters)
        for i in range(len(holder)):
            if hyperparameters[i] == "batch_size":
                self.batch_size += 10
            elif hyperparameters[i] == "learning_rate":
                self.learning_rate *= self.variation_factor
            #elif hyperparameters[i] == "shift_factor":
            #    self.shift_factor *= self.variation_factor
            elif hyperparameters[i] == "bias_spread":
                self.bias_spread += 10
            else:
                self.dimensions[int(hyperparameters[i])] += 1
            holder.pop(i)   
            if i == len(holder):
                break     
        self.parameters = Parameters.from_dimensions(self.dimensions, False, self.bias_spread, self.seed)

    def infer_dataset(self, data: List[Tuple[List[float], int]]) -> float:
        success = 0
        wu = 0
        print("hu")
        for datum in data:
            classification = self.infer(datum[0])
            if classification.index(max(classification)) == datum[1]:
                success += 1
            wu += 1
            if wu % 1000 == 0:
                print(wu)
        return success / len(data)

    def __save(self) -> dict:
        exception = None
        model_dict = deepcopy(self.__dict__)
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
        
    def load(self, path: str) -> dict:
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