class MathOperations:
    #TODO in the future remove all exception raising
    #TODO define returns and parameters better
    #TODO add softmax and cross entropy¨
    #TODO refactor vector addition and matrix_x_vector
    def matrix_x_vector(matrix, vector):
        if len(matrix[0]) != len(vector):
            raise ValueError("Length is not compatible")

        result = []
        for i in range(len(matrix[0])):
            partialSum = 0
            for j in range(len(matrix)):
                partialSum += matrix[i][j] * vector[j]
            result.append(partialSum)
        return result

    def vector_addition(u, v):
        if len(u) != len(v):
            raise ValueError("Vectors must have the same length")
        resultVector = []
        for i in range(len(u)):
            resultVector.append(u[i] + v[i])
        return resultVector

    def factorial(n):
        fact = 1
        for i in range(1, n + 1):
            fact *= i
        return fact
    def exp(x):
        e = 0
        for i in range(10): # accuracy - keep the number even
            e += x**i/MathOperations.factorial(i)
        return e
    def sigmoid(x): # 5 digit accuracy
        if(x >= 0):
            return 1-1/(MathOperations.exp(x)+1)
        else:
            return 1/(MathOperations.exp(-x)+1)
    def cost_function(answer_vector, correct_vector):
        """
        answer_vector (forward pass): last layer of results (0-9) where for example [0.1, 0.45, ...], uncorrected result of nn, correct_vector: ideal anwser (pure 1)
        """
        if (len(answer_vector) == len(correct_vector)):
            result = 0
            for i in range(len(answer_vector)):
                difference = correct_vector[i] - answer_vector[i]
                result += difference * difference
            return result
        return ValueError
    def soft_max(vector):
        sum = 0
        for element in vector:
            sum += MathOperations.exp(element)
        for i in range(len(vector)):
            vector[i] = MathOperations.exp(vector[i]) / sum

        return vector
    def taylor_ln(x, n):
        a = 0.55
        result = -0.597837000756
        for k in range(1, n + 1):
            result += ((-1) * (k - 1)) * ((x - a)**k) / (k * a ** k)
        return result

    def cross_entropy(vector, index):
        vector[index]


