from datetime import datetime as Date
from data_manager import Loader
from objects.session import Session
from objects.model import Model
import random as rnd
import hashlib
import time
import os

class CLI():
    #TODO test all methods and debug
    def load() -> list:
        result = {}
        while True:
            print("Please input whether you want to load a model (L) or load a folder with models (LM)")
            start = input().strip()
            if start == "(L)" or start == "L":
                result = {}
                while True:
                    print("Input model path")
                    path = input().strip()
                    result = Model(path)
                    if result["status"]:
                        print("Model loaded")
                        break
                    print(f"There was a problem with loading that model:\nException: {result["exception"]}")
                return [result["model"]]
            elif start == "(LM)" or start == "LM":
                while True:
                    print("Input folder path")
                    path = ""
                    while True:
                        print("Do you wish to use default path? (Y/N)")
                        default = input()
                        if default == "Y" or default == "(Y)":
                            path = os.path.join(os.path.dirname(__file__), "..\\models")
                            break
                        elif default == "N" or default == "(N)":
                            print("Input folder path")
                            path = input().strip()
                            break
                        print(f"Not one of the options. Your answer: {default}")
                    models = []
                    try:
                        if os.path.exists(path):
                            for DirEntry in os.scandir(path):
                                result = Model(str(DirEntry))
                                if result["status"]:
                                    models.append(result["model"])
                                else:
                                    raise result["exception"]
                            break
                        print(f"Not a valid path. inputed path: {path}")
                    except Exception as e:
                        print(f"There was a problem with loading that folder\nException: {e}")
                return models
            print(f"Not one of the options. Your answer: {start}")

    def create() -> Model:
        result = {}
        hidden_layers = []
        bias_spread = []
        seed = float('nan')
        data = Loader.load_data() #TODO factor in exception handling when it is created
        while True:
            print("Do you wish to use default initialization? (Y/N)")
            default = input().strip()
            if default == "Y" or default == "(Y)":
                model = Model()
                if model.id == None:
                    print(f"There was a problem with creating the default initialization:\nException: {result["exception"]}")
                    return []
                model.fit(data)
                return model
            elif default == "N" or default == "(N)":
                while True:
                    print("Define layers, input and output layers are fixed. Example input: 16,16")
                    print("If you wish to use default input (X)")
                    layers = input().strip()
                    layers_list = []
                    if layers == "(X)" or layers == "X":
                        layers_list = [16, 16]
                        break
                    try:
                        for element in layers.split(","):
                            dimension = float(element.strip())
                            layers_list.append(dimension)
                    except TypeError:
                        print(f"Invalid input please input only ints. Your input {layers}")
                    if layers_list != []:
                        break
                while True:
                    print("Define bias spread. Example: input is 10 -> minumum bias: -10 maximum bias: 10")
                    print("If you wish to use default input (X)")
                    bias_spread = input().strip()
                    if bias_spread == "(X)" or bias_spread == "X":
                        bias_spread = 10
                        break
                    if bias_spread.isdigit():
                        bias_spread = int(bias_spread)
                        break
                    print(f"Inputed bias spread is not a number. Inputed value: {bias_spread}")
                print("Define random seed for the new model")
                print("If you wish to use default input (X)")
                seed = input().strip()
                result = Model(data, hidden_layers, bias_spread, seed)
                if not result["status"]:
                    print(f"There was a problem with creating the defined that model:\nException: {result["exception"]}")
                    return Model()
                return result["model"]
            print(f"Not one of the options. Your answer: {default}")

    def infer(model: Model) -> Session:
        while True:
            rnd.seed("0")
            example = [rnd.random() for _ in range(model.dimensions[0])]
            print(f"Input the the first layer.\nExample {example}\nMatch this length {model.dimensions[0]}")
            print("If you wish to use the example input (X)")
            layer = input().strip()
            layer_list = []
            if layer == "(X)" or layer == "X":
                layer_list = example
                break
            layer_split = layer.split(",")
            if len(layer_split) == len(model.dimensions[0]):
                try:
                    for element in layer_split:
                        activation = float(element.strip())
                        layer_list.append(activation)
                    break
                except TypeError:
                    print(f"Invalid input please input only floats. Your input {layer}")
            else:
                print(f"Length not matching. Your input: {layer} Length: {len(layer_split)}")
        result = model.forward(layer_list, model["weights"], model["biases"])
        print(f"Model {model["name"]} with {model["dimensions"]} at {model["path"]}\n\tResults: {result["session"].results}")
        if not result["status"]:
            print(f"There was a problem with saving this session:\nException: {result["exception"]}")
        return result["session"]

    def main(): #main for the CLI. Surrogate main for the whole project
        print("CLI app started") 
        print("Add models:")
        sessions = []
        while True:
            models = []
            selecting = True
            while selecting:
                print("Do you want create models (R) load models (L) or remove added models (E)")
                start = input()
                if start == "(R)" or start == "R":
                    model = CLI.create()
                    if model.dimensions != None:
                        models.append()
                    else:
                        print("Model not addded")
                elif start == "(L)" or start == "L":
                    models += CLI.load()
                elif start == "(E)" or start == "E":
                    while True:
                        print(f"Input the names or ids of the models you want remove. Example input: {hashlib.md5(f"{"ITG"}+{Date.time}".encode()).hexdigest()}")
                        id = input()
                        if len(id) == 32:
                            for i in range(len(models)):
                                if models[i].id == id:
                                    models.pop(i)
                                    break
                            while True:
                                print("Remove more (R) or stop (S)")
                                remove = input()
                                if remove == "(R)" or remove == "R":
                                    continue
                                elif remove == "(S)" or remove == "S":
                                    break
                                print(f"Not one of the options. Your answer: {remove}")
                        print(f"Invalid id. MD5 digest is 32 characters long. Your answer: {id} Length: {len(id)}")
                else:
                    print(f"Not one of the options. Your answer: {start}")
                print("Loaded models:")
                for model in models:
                    print(f"\tModel {model.id} with {model.dimensions} at {model.path}")
                while True:
                    print("Do you wish to continue adding/removing models (O) or stop (S)")
                    end = input()
                    if end == "(O)" or end == "O":
                        break
                    elif end == "(S)" or end == "S":
                        selecting = False
                        break
                    print(f"Not one of the options. Your answer: {end}")
            if models == []:
                print("No models added...")
                continue
            while True:
                print("Do you want to infer with the models (Y/N)")
                infer = input()
                if infer == "(Y)" or infer == "Y":
                    for model in models:
                        sessions.append(CLI.infer(model))
                    break
                if infer == "(N)" or infer == "N":
                    break
            print("Run program again? (Y/N)")
            again = input()
            if again == "(N)" or again == "N":
                print("Sessions:\n" + sessions)
                break
        print("Ending program ...")
        time.sleep(2)

if __name__ == "__main__":
    CLI.main()