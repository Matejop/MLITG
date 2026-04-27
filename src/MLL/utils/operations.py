from typing import List
import math

#TODO add important parameters/constants to global config

class Operations:
    """
    Collection of mathematical utilities used throughout the neural network.

    Includes:
    - Vector and matrix operations
    - Activation functions and their derivatives
    - Loss functions
    - Helper utilities

    All methods are stateless and operate purely on input values.
    """
    #vector matrix operations    
    def vector_matrix_product(u: List[float], m: List[List[float]], transpose: bool = False) -> List[float]:
        """
        Computes the product of a vector and a matrix.

        Args:
            u (List[float]): Input vector.
            m (List[List[float]]): Matrix.
            transpose (bool): If True, treats the matrix as transposed.

        Returns:
            List[float]: Resulting vector.

        Raises:
            Exception:
                - If dimensions are incompatible for multiplication.

        Notes:
            - If transpose=False: computes u · m
            - If transpose=True: computes u · m^T
        """
        result = []
        if len(u) != len(m) and not transpose:
            raise Exception(f"Width of vector and height of matrix must be same: Width of vector u: {len(u)}. Height of matrix m: {len(m)}")
        elif len(u) != len(m[0]) and transpose:
            raise Exception(f"Width of vector and width of matrix must be same: Width of vector u: {len(u)} Width of Matrix m: {len(m[0])}")
        if transpose == False:
            for i in range(len(m[0])):
                partialSum = 0
                for j in range(len(m)):
                    partialSum += u[j] * m[j][i]
                result.append(partialSum)
        else:
            for i in range(len(m)):
                partialSum = 0
                for j in range(len(m[0])):
                    partialSum += u[j] * m[i][j]
                result.append(partialSum)
        return result

    #vector only operations

    def vector_addition(u: List[float], v: List[float]) -> List[float]:
        """
        Performs element-wise addition of two vectors.

        Args:
            u (List[float]): First vector.
            v (List[float]): Second vector.

        Returns:
            List[float]: Resulting vector.

        Raises:
            Exception: If vectors have different lengths.
        """
        if len(u) != len(v):
            raise Exception(f"Length of vectors must be the same: Length of u: {len(u)}. Length of v: {len(v)}")
        return [u[i] + v[i] for i in range(len(u))]
        
    #activation functions    
    
    def softmax(u: List[float]) -> List[float]:
        """
        Applies the softmax function to a vector.

        Converts raw scores (logits) into a probability distribution.

        Args:
            u (List[float]): Input vector.

        Returns:
            List[float]: Probability distribution (sums to 1).

        Notes:
            - Uses numerical stabilization by subtracting max(u).
        """
        max_val = max(u)
        sum_exp = 0
        exp_vals = []
        for i in range(len(u)):
            sum_exp += math.exp(u[i] - max_val)
            exp_vals.append(math.exp(u[i] - max_val))  
        return [x / sum_exp for x in exp_vals]

    def get_activation(input: float, type: str) -> float:
        """
        Applies the specified activation function.

        Args:
            input (float): Input value.
            type (str): Activation type ("relu", "sigmoid", "tanh").

        Returns:
            float: Activated value.

        Raises:
            Exception: If activation type is not supported.
        """
        if type == "relu":
            return Operations.relu(input)
        elif type == "sigmoid":
            return Operations.sigmoid(input)
        elif type == "tanh":
            return Operations.tanh()    
        raise Exception(f"Activation {type} is not implemented")

    def get_activation_derivative(input: float, type: str) -> float:
        """
        Computes the derivative of the specified activation function.

        Args:
            input (float): Input value (typically activation output).
            type (str): Activation type ("relu", "sigmoid", "tanh").

        Returns:
            float: Derivative value.

        Raises:
            Exception: If activation type is not supported.
        """
        if type == "relu":
            return Operations.relu_derivation(input)
        elif type == "sigmoid":
            return Operations.sigmoid_derivation(input)
        elif type == "tanh":
            return Operations.tanh_derivation(input)
        raise Exception(f"Activation {type} is not implemented")

    def relu(input: float):
        """
        ReLU activation function.

        Returns:
            float: max(0, input)
        """
        return max(0, input)
    
    def relu_derivation(input: float):
        """
        Derivative of ReLU.

        Args:
            input (float): Expected to be ReLU output.

        Returns:
            float: 0 if input <= 0 else 1
        """
        return 0 if max(0, input) == 0 else 1 
    
    def sigmoid(input: float):
        """
        Sigmoid activation function.

        Returns:
            float: Value in range (0, 1)
        """
        return 1 / (1 + math.exp(-input))
    
    def sigmoid_derivation(input: float):
        """
        Derivative of sigmoid function.

        Args:
            input (float): Expected to be sigmoid output.

        Returns:
            float: input * (1 - input)
        """
        return input * (1 - input)

    def tanh(input: float):
        """
        Hyperbolic tangent activation function.

        Returns:
            float: Value in range (-1, 1)
        """
        return (math.exp(input) - math.exp(-input)) / (math.exp(input) + math.exp(-input))
    
    def tanh_derivation(input: float):
        """
        Derivative of tanh function.

        Returns:
            float: 1 - tanh(input)^2

        Note:
            Current implementation contains an error (missing function call).
        """
        return 1 - Operations.tanh(input)**2  
    
    #loss function
    
    def cross_entropy(u: List[float], index: int) -> list:
        """
        Computes cross-entropy loss for a single prediction.

        Args:
            u (List[float]): Predicted probability distribution.
            index (int): Index of the correct class.

        Returns:
            float: Cross-entropy loss.

        Notes:
            - Uses epsilon to avoid log(0).
        """
        eps = 1e-12
        return -math.log(u[index] + eps)
    
    #other operations
    
    def try_parse(data: str) -> int | None:
        """
        Attempts to parse a string into an integer.

        Args:
            data (str): Input string.

        Returns:
            int | None:
                Parsed integer if successful, otherwise None.
        """
        try:
            return int(data)
        except:
            return None