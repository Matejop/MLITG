import numpy as np
from model import Model
import orjson
import os

def to_matrices(model: dict) -> dict:
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
            
def to_vectors(model: dict) -> dict:
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

def save_model(model: Model) -> dict:
    exception = None
    try:
        dir_path = os.path.join(os.path.dirname(__file__), "..\\..\\models")
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        model.path = f"{dir_path}\\{model["id"]}_{model["created_at"]}.json"
        model_dir = {
            "id": model.id,
            "created_at": model.created_at,
            "path": model.path,
            "dimensions": model.dimensions,
            "biases": model.biases,
            "weights": model.weights
        }
        with open(model.path, "w") as f:
            f.write(orjson.dumps(
                model_dir, 
                option=orjson.OPT_INDENT_2).decode()
            )
    except Exception as e:
        exception = e
    return {
        "status": True if exception is None else False,
        "model": model if exception is None else {},
        "exception": exception
    }
    
def load_model(path: str) -> dict:
    exception = None
    try:
        model = dict(orjson.loads(open(path).read()))
    except Exception as e:
        exception = e
    return {
        "status": True if exception is None else False,
        "model": model if exception is None else {},
        "exception": exception
    }

def load_models(path = None) -> dict:
        exception = None
        try:
            models = []
            dir_path = os.path.join(os.path.dirname(__file__), "..\\..\\models") if path == None else path
            for DirEntry in os.scandir(dir_path):
                result = load_model(DirEntry)
                if result["status"]:
                    models.append(result["model"])
                else:
                    raise Exception(result["exception"])
        except Exception as e:
            exception = e
        return {
            "status": True if exception is None else False,
            "models": models if exception is None else [],
            "exception": exception
        }
def get_layer(layer: str, element_type: type) -> list:
    layer_list = []
    for i, x in enumerate(layer.split(",")):
        x = x.strip()
        try:
            float(x)
            layer_list = []
            break
        except ValueError:
            print(f"Layer {i + 1} is not {element_type}. Layer {i + 1} inputed value: {layer} with type {x_type}")
        if x != element_type:
            print(f"Layer {i + 1} is not {element_type}. Layer {i + 1} inputed value: {layer} with type {x_type}")
            layer_list = []
            break
        layer_list.append(element_type(layer))
    return layer_list