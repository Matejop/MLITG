from datetime import datetime as Date
from typing import List
import hashlib
import orjson
import os

FOLDER_PATH = os.path.join(os.path.dirname(__file__), "../sessions")

class Session(): 
    #Session saving will be moved to a different branch once a model is trained

    def __init__(self):
        self.id = None
        self.training = None
        self.created_at = self.created_at = Date.now().__str__() 
        self.path = None
        self.model_id = None
        self.activations = None
        self.zeds = None
        self.final_layer = None

    @classmethod
    def from_training(cls, activations: List[List[float]], zeds: List[List[float]], model_id: str):
        self = cls()
        del self.final_layer
        self.training = True 
        self.id = hashlib.md5(f"ITGses+{self.created_at}".encode()).hexdigest()
        self.path = os.path.join(FOLDER_PATH, f"{self.id}.json")
        self.model_id = model_id
        self.activations = activations
        self.zeds = zeds
        result = self.__save()
        if not result["status"]:
            print(result["exception"])
            self = Session()
        return self
    
    @classmethod
    def from_inference(cls, final_layer: List[float], model_id: str):
        self = cls()
        del self.activations
        del self.zeds
        self.training = False
        self.id = hashlib.md5(f"ITGses+{self.created_at}".encode()).hexdigest()
        self.path = self.path = f"{FOLDER_PATH}/{self.id}.json"
        self.model_id = model_id
        self.final_layer = final_layer
        result = self.__save()
        if not result["status"]:
            print(result["exception"])
            self = Session()
        return self

    def __save(self) -> dict:
        exception = None
        try:
            if not os.path.exists(self.path):
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as f:
                print(self.__dict__)
                f.write(orjson.dumps(
                    self.__dict__,
                    option=orjson.OPT_INDENT_2).decode()
                )
        except Exception as e:
            exception = e
        return {
            "status": True if exception is None else False,
            "exception": exception
        }