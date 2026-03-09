from typing import List

class MathOperations:
    #TODO add important parameters/constants to global config

    def matrix_x_vector(matrix: List[List[float]], vector: List[float]) -> List[float]:
        result = []
        for i in range(len(matrix)):
            partialSum = 0
            for j in range(len(matrix[0])):
                partialSum += matrix[i][j] * vector[j]
            result.append(partialSum)
        return result

    def vector_addition(u: List[float], v: List[float]) -> List[float]:
        for i in range(len(u)):
            u[i] += v[i]
        return u

    def factorial(input: int) -> int:
        fact = 1
        for i in range(1, input + 1):
            fact *= i
        return fact
    
    def taylor_exp(input: int) -> float:
        e = 0
        for i in range(10): # accuracy - keep the number even
            e += input**i/MathOperations.factorial(i)
        print(e)
        print("taylor_exp")
        return e 
        
    def taylor_ln(input: float, sum_bound = 200) -> float:
        a = 0.55 #TODO add explaination for picking 0.55 not only that it works
        result = -0.597837000756 #offset constant = ln(a)
        for k in range(1, sum_bound + 1):
            result += ((-1)**(k - 1)) * ((input - a)**k) / (k * a ** k)
        return result
    
    def sigmoid(input: float) -> float: # 5 digit accuracy
        if(input >= 0):
            print(1-1/(MathOperations.taylor_exp(input)+1))
            print("sigmoid")
            return 1-1/(MathOperations.taylor_exp(input)+1)
        else:
            print(1/(MathOperations.taylor_exp(-input)+1))
            print("sigmoid")
            return 1/(MathOperations.taylor_exp(-input)+1)
        
    def softmax(vector: List[float]) -> List[float]:
        sum = 0
        for element in vector:
            sum += MathOperations.taylor_exp(element)
        for i in range(len(vector)):
            vector[i] = MathOperations.taylor_exp(vector[i]) / sum
        return vector
    
    def cross_entropy(vector: List[float], index: int) -> float:
        return -1 * MathOperations.taylor_ln(vector[index])