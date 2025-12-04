import numpy as np

def to_matrices(model: dict) -> dict: #TODO is redundant consider removing
        dimensions = model["dimensions"]
        biases = np.empty(len(dimensions) - 1, dtype=np.ndarray)
        weights = np.empty(len(dimensions) - 1, dtype=np.ndarray)
        biases_index = 0
        weights_index = 0
        for i in range(1, len(dimensions)):
            biases[i - 1] = model["biases"][biases_index : biases_index + dimensions[i]]
            weights[i - 1] = np.empty((dimensions[i], dimensions[i - 1]))
            for j in range(1, dimensions[i]):
                for k in range(1, dimensions[i - 1]):
                    weights[i - 1][j][k] = model["weights"][weights_index]
                    weights_index += 1
            biases_index += dimensions[i]
        model["biases"] = biases
        model["weights"] = weights
        return model
            
def to_vectors(model: dict) -> dict: #TODO is redundant consider removing
    biases = []
    weights = []
    for i in range(1, len(model["dimensions"])):
        for j in range(len(model["weights"][i - 1])):
            for k in range(len(model["weights"][i - 1][j])):
                weights.append(model["weights"][i - 1][j][k])
        biases += list(model["biases"][i - 1])

    model["biases"] = np.array(biases)
    model["weights"] = np.array(weights)
    return model