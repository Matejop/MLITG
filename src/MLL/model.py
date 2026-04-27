from MLL.utils.operations import Operations
from MLL.utils.input_validation import InputValidation
from MLL.utils.data_manager import DataManager
from datetime import datetime as Date
from typing import List, Tuple
from copy import deepcopy
import random as rnd
import hashlib
import orjson
import math
import os

class Model():   
    """
    Core neural network model used for training and classification.

    This class implements a fully connected feedforward neural network with:
    - Configurable layer dimensions
    - Multiple activation functions
    - Mini-batch gradient descent training
    - 
    - Cross-entropy loss with softmax output

    The model supports multiple training trials and selects the best-performing
    parameters based on loss.

    Attributes:
        id (str): Unique identifier for the model.
        created_at (str): Timestamp when the model was created.
        path (str): File path where the model is saved.
        dimensions (List[int]): Sizes of each layer (input → output).
        activation (str): Activation function used in hidden layers.
        trials (int): Number of training trials.
        epochs (int): Number of epochs per trial.
        batch_size (int): Number of samples per batch.
        learning_rate (float): Gradient descent step size.
        seed (str): Random seed used for initialization.
        biases (List[List[float]]): Bias vectors for each layer.
        weights (List[List[List[float]]]): Weight matrices for each layer.
    """
    
    def __init__(self):
        self.id = None
        self.created_at = None     
        self.path = None  
        self.dimensions = None
        self.activation = None
        self.trials = None
        self.epochs = None
        self.batch_size = None
        self.learning_rate = None
        self.seed = None
        self.biases = None
        self.weights = None

    @classmethod
    def new(cls, dimensions: List[int], activation: str = "relu", trials: int = 3, epochs: int = 10, batch_size: int = 1000, learning_rate: float = 0.01):
        """
        Factory method to create a new Model instance with validated parameters.

        Args:
            dimensions (List[int]): Network architecture (input → hidden → output).
            activation (str): Activation function for hidden layers.
            trials (int): Number of independent training runs.
            epochs (int): Number of epochs per trial.
            batch_size (int): Size of training batches.
            learning_rate (float): Learning rate for gradient descent.

        Returns:
            Model: Initialized model instance.

        Raises:
            Exception: If input validation fails.
        """
        error_message = InputValidation.check_model_input(dimensions, activation, trials, epochs, batch_size, learning_rate)
        if error_message != "":
            raise Exception(f"An exception occured while creating a model: {error_message}")
        time_now = Date.now().__str__()
        model = cls()
        model.id = hashlib.md5(f"{"ITG"}+{time_now}".encode()).hexdigest()
        model.created_at = time_now     
        model.dimensions = dimensions
        model.activation = activation
        model.trials = trials
        model.epochs = epochs
        model.batch_size = batch_size
        model.learning_rate = learning_rate
        return model
    
    def fit(self, x: List[List[float]], y: List[List[int]], seed: str = None):
        """
        Trains the model on the provided dataset.

        Training is performed over multiple trials. The model retains the
        weights and biases from the trial with the lowest loss.

        Args:
            x (List[List[float]]): Input features.
            y (List[List[int]]): One-hot encoded labels.
            seed (str, optional): Seed for reproducibility.

        Raises:
            Exception: If data is invalid or training fails.
        """
        try:
            error_message = InputValidation.check_data_len(x, y)
            if error_message != "":
                raise Exception(error_message)
            best_biases = []
            best_weights = []            
            smallest_loss = None
            batched_data = DataManager.batch_data(x, y, self.batch_size)
            for trial in range(self.trials):                
                print(f"Trial: {trial + 1}")
                self.__initialize_parameters(seed)    
                trial_loss = 0.0
                for epoch in range(self.epochs):                       
                    epoch_loss = self.__train(batched_data)
                    trial_loss = epoch_loss / len(x) 
                    print(f"Epoch: {epoch + 1}\nAverage loss: {epoch_loss / len(x)}")
                if smallest_loss == None:
                    smallest_loss = trial_loss
                    best_biases = deepcopy(self.biases)
                    best_weights = deepcopy(self.weights)
                elif trial_loss < smallest_loss:
                    smallest_loss = trial_loss
                    best_biases = deepcopy(self.biases)
                    best_weights = deepcopy(self.weights)
            self.biases = best_biases
            self.weights = best_weights
        except Exception as e:
            raise Exception(f"An exception occured while fitting to data: Exception: {e}")

    def classify(self, activations: List[float], layer_index: int = 0) -> List[float]:
        """
        Performs forward propagation to classify a single input.

        This method is recursive and should be called without specifying
        `layer_index`.

        Args:
            activations (List[float]): Input feature vector.
            layer_index (int): Internal recursion index.

        Returns:
            List[float]: Probability distribution over output classes.

        Raises:
            Exception: If input dimensions do not match model architecture.
        """
        try:
            if len(activations) != self.dimensions[layer_index]:
                raise Exception(f"Input has mismatching length. Length of input: {len(activations)}. Size of dimension {layer_index}: {self.dimensions[layer_index]}")
            zeds = Operations.vector_addition(Operations.vector_matrix_product(activations, self.weights[layer_index]), self.biases[layer_index])
            if layer_index < len(self.weights) - 1:
                layer_index += 1
                return self.classify([Operations.get_activation(x, self.activation) for x in zeds], layer_index)
            else:
                return Operations.softmax(zeds)
        except Exception as e:
            raise Exception(f"An exception occured while a model was classifying: Model id: {self.id}. Input: {activations}. Exception: {e}")      

    #infrastructure

    def __initialize_parameters(self, seed: str = None):
        self.seed = Date.now().__str__() if seed == None else seed
        rnd.seed(self.seed)
        self.biases = []
        self.weights = []
        #xavier initialization
        limit = math.sqrt(6 / (self.dimensions[0] + self.dimensions[-1]))   
        for i in range(1, len(self.dimensions)):
            self.biases.append([])
            self.weights.append([])    
            #xavier initialization        
            limit = math.sqrt(6 / (self.dimensions[i - 1] + self.dimensions[i]))   
            for j in range(self.dimensions[i - 1]):                                     
                self.weights[i - 1].append([])
                for _ in range(self.dimensions[i]):                                       
                    self.weights[i - 1][j].append(rnd.uniform(-limit, limit))
            for j in range(self.dimensions[i]):
                bias_value = 0 if self.activation != "relu" else 0.01
                self.biases[i - 1].append(bias_value) 

    def __train(self, batched_data: List[List[Tuple[List[int], List[float]]]]) -> float:
        loss = 0.0
        for batch in batched_data:                 
            for i, datum in enumerate(batch): 
                if len(datum[0]) != self.dimensions[0]:
                    raise Exception(f"Inputed X at index {i} does not have the same length as input layer. X of datum: {len(datum[0])} Input layer size: {self.dimensions[0]}")
                if len(datum[1]) != self.dimensions[-1]:
                    raise Exception(f"Inputed Y at index {i} does not have the same length as output layer. Y of datum: {len(datum[1])} Output layer size: {self.dimensions[-1]}")
                activations = self.__forward(datum[0])  
                answer_index = None
                for j in range(len(datum[1])):
                    if datum[1][j] == 1:
                        answer_index = j
                        break
                loss += Operations.cross_entropy(activations[-1], answer_index)
                self.__backward(activations, answer_index)    
        return loss

    def __forward(self, input) -> List[List[float]]:
        #getting activations by passing forward
        activations = []
        activations.append(input) 
        for l in range(len(self.weights)):
            zeds = Operations.vector_addition(
                Operations.vector_matrix_product(
                    activations[l], 
                    self.weights[l]), 
                self.biases[l]
            )
            if l < len(self.weights) - 1:
                activations.append([Operations.get_activation(x, self.activation) for x in zeds])
            else:
                activations.append(Operations.softmax(zeds))
        return activations
    
    def __backward(self, activations: List[List[float]], answer_index: List[int]):
        #backpropagation - gradient descent
        delta = deepcopy(activations[-1])
        delta[answer_index] -= 1
        for l in range(1, len(self.weights)):  
            for i in range(len(self.biases[-l])):
                self.biases[-l][i] -= delta[i] * self.learning_rate
            for i, delta_element in enumerate(delta):
                for j, activation in enumerate(activations[-(l + 1)]):
                    self.weights[-l][j][i] -= delta_element * activation * self.learning_rate
            if l + 1 < len(self.weights):
                delta = Operations.vector_matrix_product(delta, self.weights[-l], True)
                delta = [
                    delta[i] 
                    * Operations.get_activation_derivative(activation, self.activation)
                    for i, activation in enumerate(activations[-(l + 1)])
                ]

    def measure_accuracy(self, x: List[List[float]], y: List[List[int]]) -> float:
        """
        Evaluates model accuracy on a dataset.

        Args:
            x (List[List[float]]): Input features.
            y (List[List[int]]): One-hot encoded labels.

        Returns:
            float: Accuracy score (0.0 to 1.0).

        Raises:
            Exception: If input data is invalid.
        """
        try:
            error_message = InputValidation.check_data_len(x, y)
            if error_message != "":
                raise Exception(error_message)
            success = 0
            for i in range(len(x)):
                if len(x[i]) != self.dimensions[0]:
                    raise Exception(f"Inputed X at index {i} does not have the same length as input layer. X of datum: {len(x[i])} Input layer size: {self.dimensions[0]}")
                if len(y[i]) != self.dimensions[-1]:
                    raise Exception(f"Inputed Y at index {i} does not have the same length as output layer. Y of datum: {len(y[i])} Output layer size: {self.dimensions[-1]}")
                classification = self.classify(x[i])
                answer_index = 0                
                max_classification_index = 0
                max_classification = classification[0]
                for j in range(len(y[i])):
                    if y[i][j] == 1:
                        answer_index = j
                    if classification[j] > max_classification:
                        max_classification = classification[j]
                        max_classification_index = j
                if max_classification_index == answer_index:
                    success += 1
            return success / len(x)
        except Exception as e:
            raise Exception(f"An exception occured while measuring accuracy of a model: Model id: {self.id}. Exception: {e}")

    def save(self, path: str = None):
        """
        Saves the model to a JSON file.

        If no path is set, a new file path is generated using the model ID.

        Args:
            path (str, optional): Directory where the model should be saved.

        Raises:
            Exception: If saving fails or file already exists.
        """
        try:
            if self.path == None:   
                model_path = os.path.join(path, f"Model_{self.id[:10]}.json")
                print(model_path)
                if os.path.exists(model_path) == True:
                    model_path = os.path.join(path, f"Model{self.id}.json")
                if os.path.exists(model_path) == True:
                    raise Exception(f"The inputed path already exists")
                self.path = model_path
            self.hash = hashlib.sha256(self).__str__()
            with open(self.path, "w") as f:
                f.write(orjson.dumps(
                    self.__dict__, 
                    option=orjson.OPT_INDENT_2).decode()
                )
        except Exception as e:
            raise Exception(f"An exception occured while saving the model to path: Model id: {self.id}. Path: {path}. Exception: {e}")
    
    @classmethod
    def load(cls, path: str):
        """
        Loads a model from a JSON file.

        Args:
            path (str): Path to the saved model file.

        Returns:
            Model: Reconstructed model instance.

        Raises:
            Exception: If loading fails.
        """
        try:
            model_dict = dict(orjson.loads(open(path).read()))
            model = cls()
            model.id = model_dict["id"]
            model.created_at = model_dict["created_at"]     
            model.path = model_dict["path"]  
            model.dimensions = model_dict["dimensions"]
            model.activation = model_dict["activation"]
            model.trials = model_dict["trials"]
            model.epochs = model_dict["epochs"]
            model.batch_size = model_dict["batch_size"]
            model.learning_rate = model_dict["learning_rate"]
            model.seed = model_dict["seed"]
            model.biases = model_dict["biases"]
            model.weights = model_dict["weights"]
            return model          
        except Exception as e:
            raise Exception(f"An exception occured while loading a model from path: Path: {path}. Exception: {e}")