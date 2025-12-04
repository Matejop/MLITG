from infrastructure.file_manager import FileManager as FM
from infrastructure.mnist import Mnist
from model import Model as Model
from pathlib import Path
import random as rnd
import numpy as np
import time
import ast

class CLI():
    #TODO test all methods and debug
    def load() -> dict:
            result = {}
            while True:
                print("Please input whether you want to load a model (L) or load multiple (LM)")
                start = input().strip()
                if start == "(L)" or start == "L":
                    while True:
                        print("Input model path")
                        path = input().strip()
                        if Path(path).exists:
                            result = FM.load(path)
                            break
                        print(f"Not a valid path. inputed path: {path}")
                    if not result["status"]:
                        print(f"There was a problem with loading that model:\nException: {result["exception"]}")
                        return []
                    return [result["model"]]
                elif start == "(LM)" or start == "LM":
                    while True:
                        print("Input folder path")
                        path = input().strip()
                        if Path(path).exists:
                            result = FM.load_folder(path)
                            break
                        print(f"Not a valid path. inputed path: {path}")
                    if not result["status"]:
                        print(f"There was a problem with loading that folder not all models were loaded:\nException: {result["exception"]}")
                    return result["models"]
                print(f"Not one of the options. Your answer: {start}")

    def create() -> dict:
        #TODO add input for training (part of Model.create())
        result = {}
        hidden_layers = []
        bias_spread = []
        while True:
            print("Do you wish to use default initialization? (Y/N)")
            default = input().strip()
            if default == "Y" or default == "(Y)":
                result = Model.create()
                if not result["status"]:
                    print(f"There was a problem with creating the default initialization:\nException: {result["exception"]}")
                    return []
                return [result["model"]]
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
                            element = float(element.strip())
                            layers_list.append(element)
                    except Exception as e:
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
                print("Define name for the new model")
                print("If you wish to use default input (X)")
                name = input().strip()
                result = Model.create("" if name == "(X)" or name == "X" else name, np.array(hidden_layers), bias_spread)
                if not result["status"]:
                    print(f"There was a problem with creating the defined that model:\nException: {result["exception"]}")
                    return []
                return [result["model"]]
            print(f"Not one of the options. Your answer: {default}")

    def infer(model: dict) -> np.ndarray:
        while True:
            print(f"Input the the first layer. Match this length {model["dimensions"][0]}\n Example: 0.54, 0.9999, 0.31252")
            print("If you wish to use random values (X)")
            layer = input().strip()
            layer_list = []
            if layer == "(X)" or layer == "X":
                layer_list = [rnd.random() for _ in range(model["dimensions"][0])]
            elif len(layer) == len(model["dimensions"][0]):
                try:
                    for element in layer.split(","):
                        element = float(element.strip())
                        layer_list.append(element)
                except Exception as e:
                    print(f"Invalid input please input only floats. Your input {layer}")
                    layer_list = []
            if layer_list != []:
                break
        output = Model.forward(layer_list, model["weights"], model["biases"])
        print(f"Model {model["name"]} with {model["dimensions"]} at {model["path"]}\n\tResults: {output}")
        result = FM.save_session(output, model["name"])
        if not result["status"]:
            print(f"There was a problem with saving the results:\nException: {result["exception"]}")
        return output

    def main(): #main for the CLI. Surrogate main for the whole project
        #TODO Recontextulize - methods updated and new added
        print("CLI app started") 
        print("Add models:")
        models = {}
        while True:
            while True:
                if len(models.keys()) > 0:
                    print("Do you want create models (R) load models (L) or remove added models (E)")
                elif len(models.keys()) == 0:
                    print("Do you want to create a models (R) or load models (L)")
                start = input()
                if start == "(R)" or start == "R":
                    models = [CLI.create()] 
                elif start == "(L)" or start == "L":
                    models = CLI.load()
                elif len(models.keys()) > 0 and start == "(E)" or start == "E":
                    #TODO add removing added models
                    #while True:
                        #print("Input the names or ids of the models you want remove. Example input: name: MyModel or id: 16")
                    print("Not implemented")
                else:
                    print(f"Not one of the options. Your answer: {start}")
                print("Loaded models:")
                for model in models:
                    print(f"\tModel {model["name"]} with {model["dimensions"]} at {model["path"]}")
                print("Do you wish to continue adding models (O) or end (E)")
                end = input()
                if end == "(O)" or end == "O":
                    continue
                elif end == "(E)" or end == "E":
                    break
                print(f"Not one of the options. Your answer: {end}")
            if models == []:
                print("No models added...")
                print("Ending program")
                break
            while True:
                print("Do you want to infer with the models (Y/N)")
                infer = input()
                if infer == "(Y)" or infer == "Y":





                if all == "(T)" or all == "T":
                    data = Mnist.load_wrapped()
                    for model in models:
                        #training function here
                        #print and save all results
                        print("Not implemented yet")
                    break
                elif all == "(I)" or all == "I":
                    for model in models:
                        CLI.infer(model)
                    break
                elif all == "(E)" or all == "E":
                    for model in models:
                        while True:
                            print(f"Train (T) or Infer (I)\nModel name: {model["name"]}")
                            each = input()
                            data = Mnist.load_wrapped()
                            if each == "(T)" or each == "T":
                                #training function here
                                #print and save all results
                                break
                            elif each == "(I)" or each == "I":
                                CLI.infer(model)
                                break
                            print(f"Not one of the options. Your answer: {each}")
                    break
            print("Run program again? (Y/N)")
            again = input()
            if again == "(N)" or again == "N":
                break
        print("All models adressed\nEnding program ...")
        time.sleep(2)

if __name__ == "__main__":
    CLI.main()