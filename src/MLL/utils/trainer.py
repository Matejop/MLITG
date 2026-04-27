from MLL.utils.operations import Operations
from MLL.utils.input_validation import InputValidation
from copy import deepcopy
from MLL.model import Model
from typing import List
import random as rnd
import MLL

#TODO exception raising and management

class Trainer():
    """
    Utility class for hyperparameter optimization of Model instances.

    Supports:
    - Grid search
    - Random search

    The Trainer explores combinations of:
    - Network depth and layer sizes
    - Batch size
    - Learning rate
    - Activation function

    Attributes:
        max_depth (int): Maximum number of layers.
        trials (int): Number of trials per configuration.
        epochs (int): Number of epochs per training run.
        keys (List[str]): Parameter keys used in search.
        start (dict): Starting values for search parameters.
        end (dict): Ending values for search parameters.
        step (dict): Step sizes or sampled values.
    """
    
    def __init__(self, activation: str, max_depth: int, trials: int, epochs: int, layer_size_start: int, layer_size_end: int, batch_size_start: int, batch_size_end: int, learning_rate_start: float, learning_rate_end: int):
        exception = InputValidation.check_trainer_input(activation, max_depth, trials, epochs, layer_size_start, layer_size_end, batch_size_start, batch_size_end, learning_rate_start, learning_rate_end)
        if exception != "":
            raise Exception(f"An exception occured while creating a Trainer: Exception: {exception}")
        self.max_depth = max_depth
        self.trials = trials
        self.epochs = epochs
        self.keys = ["batch_size", "learning_rate", "activation"]
        self.start = {
                "batch_size": batch_size_start,
                "learning_rate": learning_rate_start,
                "activation": 0 if activation == "all" else MLL.activation_functions.index(activation)
            }
        self.end = {
                "batch_size": batch_size_end,
                "learning_rate": learning_rate_end,
                "activation": len(MLL.activation_functions) - 1 if activation == "all" else MLL.activation_functions.index(activation)
            }
        self.step={}
        for i in range(max_depth - 2):
            self.keys.append(str(i))
            self.start[str(i)] = layer_size_start
            self.end[str(i)] = layer_size_end

    def grid_search(self, training_x: List[List[float]], training_y: List[List[float]], validation_x: List[List[float]], validation_y: List[List[float]], layer_size_step: int = 10, batch_size_step: int = 100, learning_rate_step: float = 0.001) -> Model:
        """
        Performs exhaustive grid search over hyperparameters.

        Iterates through combinations of parameters and selects the model
        with the highest validation accuracy.

        Args:
            training_x (List[List[float]]): Training inputs.
            training_y (List[List[float]]): Training labels.
            validation_x (List[List[float]]): Validation inputs.
            validation_y (List[List[float]]): Validation labels.
            layer_size_step (int): Step size for layer dimensions.
            batch_size_step (int): Step size for batch size.
            learning_rate_step (float): Step size for learning rate.

        Returns:
            Model: Best-performing model.

        Raises:
            Exception: If validation fails or training errors occur.
        """
        try:  
            exception = InputValidation.check_gridsearch_input(training_x, training_y, validation_x, validation_y, layer_size_step, batch_size_step, learning_rate_step)
            if exception != "":
                raise exception
            print("Grid search Trainer running") 
            self.step = {
                "batch_size": batch_size_step,
                "learning_rate": learning_rate_step,
                "activation": 1
            }
            start_layers = []
            for i in range(self.max_depth - 2):
                self.step[str(i)] = layer_size_step
                start_layers.append(self.start[str(i)])
            best_model = None
            best_accuracy = -1.0
            for i in range(len(self.keys) - 1, -1, -1):
                for j in range(len(self.keys) - i):
                    key_combination = self.keys[:i]
                    key_combination.append(self.keys[i + j])
                    varied_values = deepcopy(self.start)
                    dimensions = []
                    if len(start_layers) > 0:
                        dimensions = [] if Operations.try_parse(key_combination[-1]) == None else start_layers[:(int(key_combination[-1]) + 1)]
                        dimensions.insert(0, len(training_x[0])) #input dimension size
                        dimensions.append(len(training_y[0])) #output dimension size
                    while (True):
                        if best_accuracy == -1.0:                            
                            changes = False 
                            for key in key_combination:
                                varied_values[key] = varied_values[key] + self.step[key]
                                if varied_values[key] > self.end[key]:
                                    varied_values[key] = self.end[key] 
                                else:
                                    if Operations.try_parse(key) != None:
                                        dimensions[int(key)] = varied_values[key]
                                    changes = True                        
                            if changes == False:
                                break               
                        model = Model.new(
                            dimensions, 
                            MLL.activation_functions[varied_values["activation"]], 
                            self.trials,
                            self.epochs, 
                            varied_values["batch_size"], 
                            varied_values["learning_rate"]
                        )
                        model.fit(training_x, training_y)
                        accuracy = model.measure_accuracy(validation_x, validation_y) 
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model = deepcopy(model)
                        print(f"Achieved accuracy {accuracy} with configuration: {varied_values}")
            return best_model
        except Exception as e:
            raise Exception(f"An exception occured while creating a model with Trainer: Exception {e}")
    
    def random_search(self, training_x: List[List[float]], training_y: List[List[float]], validation_x: List[List[float]], validation_y: List[List[float]], iterations: int = 10) -> Model:
        """
        Performs random search over hyperparameters.

        Randomly samples configurations within defined bounds and selects
        the best-performing model based on validation accuracy.

        Args:
            training_x (List[List[float]]): Training inputs.
            training_y (List[List[float]]): Training labels.
            validation_x (List[List[float]]): Validation inputs.
            validation_y (List[List[float]]): Validation labels.
            iterations (int): Number of random configurations to test.

        Returns:
            Model: Best-performing model.

        Raises:
            Exception: If validation fails.
        """
        validation_result = InputValidation.check_rndsearch_input(training_x, training_y, validation_x, validation_y, iterations)
        if validation_result != "":
            raise Exception(f"An exception occured while creating a model with Trainer: Exception {validation_result}")
        best_accuracy = -1.0
        best_model = None
        print("Random search Trainer running") 
        for iteration in range(iterations):
            print(f"Iteration: {iteration + 1}:")
            rnd.seed("search")
            dimensions = []
            for key in self.keys:
                if key == "learning_rate":
                    self.step[key] = rnd.uniform(self.start[key], self.end[key])
                else:
                    self.step[key] = rnd.randint(self.start[key], self.end[key])
                    if Operations.try_parse(key) != None:
                        dimensions.append(self.step[key])
            dimensions.insert(0, len(training_x[0])) #input dimension size
            dimensions.append(len(training_y[0])) #output dimension size
            model = Model.new(
                dimensions, 
                MLL.activation_functions[self.step["activation"]], 
                self.trials,
                self.epochs, 
                self.step["batch_size"], 
                self.step["learning_rate"]
            )
            model.fit(training_x, training_y)
            accuracy = model.measure_accuracy(validation_x, validation_y) 
            if accuracy > best_accuracy:    
                best_accuracy = accuracy
                best_model = deepcopy(model)
            print(f"Achieved accuracy {accuracy} with configuration: {self.step}")
        return best_model
