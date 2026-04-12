from typing import List, Tuple
from objects.frac import Frac
from copy import deepcopy
import math
#from objects.parameters import Parameters


class MathOperations:
    #TODO add important parameters/constants to global config

    #vector matrix operations

    #def update_matrix(u: List[float], m: List[List[float]], t_map: List[List[Tuple[int, int]]] = None) -> List[List[float]]:
    #    if t_map == None:
    #        for i in range(len(m)):
    #            for j in range(len(m[0])):
    #                m[i][j] *= u[j]
    #    else:
    #        for i in range(len(t_map)):
    #            for j in range(len(t_map[0])):
    #                x, y = t_map[i][j]
    #                m[x][y] *= u[j] 
    #    return m

    #def update_matrix(m: List[List[float]], u: List[float], v: List[float]):
    #    for i in range(len(u)):
    #        for j in range(len(v)):
    #            m[i][j] += u[i] * v[j]
    #    return m

    
    def vector_matrix_product(u: List[float], m: List[List[float]], transpose: bool = False) -> List[float]:
        result = []
        if not transpose:
            for i in range(len(m[0])):
                partialSum = 0
                for j in range(len(m)):                    
                    #print(u[j])
                    #print(m[j][i])
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

    #def vector_scalar_product(u: list, scalar: float):
        #for i in range(len(u)):
        #    u[i] *= scalar
    #    return [u[i] * scalar for i in range(len(u))]

    def vector_addition(u: List[float], v: List[float]) -> List[float]:
        #for i in range(len(u)):
        #    u[i] += v[i]
        return [u[i] + v[i] for i in range(len(u))]
    
    def softmax(u: List[float]) -> List[float]:
        max_val = max(u)
        exp_vals = [math.exp(x - max_val) for x in u]
        sum_exp = sum(exp_vals)
        return [x / sum_exp for x in exp_vals]
        #sum = 0
        #for element in u:
        #    sum += MathOperations.taylor_exp(element)
        #for i in range(len(u)):
        #    u[i] = MathOperations.taylor_exp(u[i]) / sum
        #return u  
    
    def cross_entropy(u: List[float], index: int) -> list:#
        #u[index] = -1 * MathOperations.taylor_ln(u[index])
        eps = 1e-12
        return -math.log(u[index] + eps)
    
    #maths operations

    #def add_frac(x: Frac, y: Frac) -> Frac:
    #    common_denominator = x.denominator * y.denominator
    #    x.numerator = (x.numerator * y.denominator) + (y.numerator * x.denominator)
    #    x.denominator = common_denominator
    #    x.reduce()
    #    return x

    #def multiply_frac(x: Frac, y: Frac) -> Frac:
    #    x.numerator *= y.numerator
    #    x.denominator *= y.denominator
    #    x.reduce()
    #    return x
        
    #def divide_frac(x: Frac, y: Frac) -> Frac:
    #    y.numerator += y.denominator
    #    y.denominator = y.numerator - y.denominator
    #    y.numerator -= y.denominator
    #    return MathOperations.multiply_frac(x, y)
    
    #def raise_frac(x: Frac, y: Frac):
            

    #def factorial(input: int) -> int:
    #    fact = 1
    #    for i in range(1, input + 1):
    #        fact *= i
    #    return fact
        
    #def taylor_exp(input: float) -> float:
    #    result = 0
    #    for i in range(100): # accuracy - keep the number even
    #        result += input**i/MathOperations.factorial(i)
    #    return result

    #def taylor_ln(input: float, sum_bound = 200) -> float:
    #    a = 0.55 #TODO add explaination for picking 0.55 not only that it works
    #    result = -0.597837000756 #offset constant = ln(a)
    #    for k in range(1, sum_bound + 1):
    #        result += ((-1)**(k - 1)) * ((input - a)**k) / (k * a ** k)
    #    return result
    
    def sigmoid(input: float) -> float:
        #if (input >= 0):
        #    return 1 - 1 / (1 + math.exp(input))
        #else:
            #return 1 / (1 + math.exp(-input))
        #return 1 / (1 + MathOperations.taylor_exp(-input))
        #print(1 / (1 + math.exp(-input)))
        return 1 / (1 + math.exp(-input))

    
    #def round(input: float, down: bool = True):
    #    rounded = 0
    #    if input * -1 < 0:
    #        if down:
    #            while rounded < input:
    #                rounded += 1
    #            if rounded != input:
    #                rounded -= 1
    #        else:
    #            while rounded < input:
    #                rounded += 1            
    #    else:
    #        if down:
    #            while rounded > input:
    #                rounded -= 1
    #            if rounded != input:
    #                rounded -= 1
    #        else:
    #            while rounded > input:
    #                rounded -= 1
    #    return rounded
    
    #def max(input: List[float]):
    #    max = 0
    #    for i in range(input):
    #        if i == 0:
    #            max = input[i]
    #        else:
    #            if input[i] > max:
    #                max = input[i]
    #    return max
    
#if __name__ == "__main__":
#    result = MathOperations.squareroot(Frac(4, 1), 2)
#    print(result.numerator, result.denominator)
#    print(result.numerator / result.denominator)
#    result = MathOperations.squareroot(Frac(2, 6), 3)
#    print(result.numerator, result.denominator)
#    print(result.numerator / result.denominator)
    #params = Parameters.super_trial()
    #vector = [5, 5]
    #print(MathOperations.vector_matrix_product(vector, params.weights[0], params.transposed_map[0]))
