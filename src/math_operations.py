from typing import List

class MathOperations:
    #TODO add important parameters/constants to global config
    #TODO remove all exception raising after Model.fit() is finished

    def matrix_x_vector(matrix: List[list], vector: list):
        if len(matrix) != len(vector):
            raise ValueError("Length is not compatible")

        result = []
        for i in range(len(matrix[0])):
            partialSum = 0
            for j in range(len(matrix)):
                partialSum += matrix[i][j] * vector[j]
            result.append(partialSum)
        return result

    def vector_addition(u: list, v: list):
        if len(u) != len(v):
            raise ValueError("Vectors must have the same length")
        for i in range(len(u)):
            u[i] += v[i]
        return u

    def factorial(input: int):
        fact = 1
        for i in range(1, input + 1):
            fact *= i
        return fact
    
    def taylor_exp(base: int):
        e = 0
        for i in range(10): # accuracy - keep the number even
            e += base**i/MathOperations.factorial(i)
        return e 
        
    def taylor_ln(input: float, sum_bound = 200):
        a = 0.55 #TODO add explaination for picking 0.55 not only that it works
        result = -0.597837000756 #offset constant = ln(a)
        for k in range(1, sum_bound + 1):
            result += ((-1)**(k - 1)) * ((input - a)**k) / (k * a ** k)
        return result
    
    def sigmoid(input: float): # 5 digit accuracy
        if(input >= 0):
            return 1-1/(MathOperations.taylor_exp(input)+1)
        else:
            return 1/(MathOperations.taylor_exp(-input)+1)
        
    def softmax(vector: list):
        sum = 0
        for element in vector:
            sum += MathOperations.taylor_exp(element)
        for i in range(len(vector)):
            vector[i] = MathOperations.taylor_exp(vector[i]) / sum
        return vector
    
    def cross_entropy(vector: list, index: int):
        return -1 * MathOperations.taylor_ln(vector[index])