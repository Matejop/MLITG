from typing import List, Tuple

class MathOperations:
    #TODO add important parameters/constants to global config

    #vector matrix operations

    def update_matrix(u: List[float], m: List[List[float]], t_map: List[List[Tuple[int, int]]] = None) -> List[List[float]]:
        if t_map == None:
            for i in range(len(m)):
                for j in range(len(m[0])):
                    m[i][j] *= u[j]
        else:
            for i in range(len(t_map)):
                for j in range(len(t_map[0])):
                    x, y = t_map[i][j]
                    m[x][y] *= u[j] 
        return m
    
    def vector_matrix_product(u: List[float], m: List[List[float]], t_map: List[List[Tuple[int, int]]] = None) -> List[float]:
        result = []
        if t_map == None:
            for i in range(len(m)):
                partialSum = 0
                for j in range(len(m[0])):
                    partialSum += m[i][j] * u[j]
                result.append(partialSum)
        else:
            for i in range(len(t_map)):
                partialSum = 0
                for j in range(len(t_map[0])):
                    x, y = t_map[i][j] 
                    partialSum += m[x][y] * u[j]
                result.append(partialSum)
        return result

    #vector only operations

    def vector_scalar_product(u: list, scalar: float):
        for i in range(len(u)):
            u[i] *= scalar
        return u

    def vector_addition(u: List[float], v: List[float]) -> List[float]:
        for i in range(len(u)):
            u[i] += v[i]
        return u

    def softmax(u: List[float]) -> List[float]:
        sum = 0
        for element in u:
            sum += MathOperations.taylor_exp(element)
        for i in range(len(u)):
            u[i] = MathOperations.taylor_exp(u[i]) / sum
        return u
    
    def cross_entropy(u: List[float], index: int) -> list:
        u[index] = -1 * MathOperations.taylor_ln(u[index])
        return u
    
    #maths operations

    def factorial(input: int) -> int:
        fact = 1
        for i in range(1, input + 1):
            fact *= i
        return fact
    
    def taylor_exp(input: int) -> float:
        e = 0
        for i in range(10): # accuracy - keep the number even
            e += input**i/MathOperations.factorial(i)
        return e 
        
    def taylor_ln(input: float, sum_bound = 200) -> float:
        a = 0.55 #TODO add explaination for picking 0.55 not only that it works
        result = -0.597837000756 #offset constant = ln(a)
        for k in range(1, sum_bound + 1):
            result += ((-1)**(k - 1)) * ((input - a)**k) / (k * a ** k)
        return result
    
    def sigmoid(input: float) -> float: # 5 digit accuracy
        if (input >= 0):
            return 1 - 1 / (MathOperations.taylor_exp(input)+1)
        else:
            return 1 / (MathOperations.taylor_exp(-input)+1)
    
    def round(input: float, down: bool = True):
        rounded = 0
        if input * -1 < 0:
            if down:
                while rounded < input:
                    rounded += 1
                if rounded != input:
                    rounded -= 1
            else:
                while rounded < input:
                    rounded += 1            
        else:
            if down:
                while rounded > input:
                    rounded -= 1
                if rounded != input:
                    rounded -= 1
            else:
                while rounded > input:
                    rounded -= 1
        return rounded
    
    def max(input: List[float]):
        max = -1
        for element in input:
            if element > max:
                max = element
        return max
