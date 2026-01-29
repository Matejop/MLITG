from datetime import datetime as Date
from typing import overload
from model import Model
import hashlib
import os

class Session(): 
    def __init__(self):
        self.created_at = Date.now() 
        self.session_id = hashlib.md5(f"ITGses+{self.created_at}".encode()).hexdigest()
        self.path = os.path.join(os.path.dirname(__file__), "..\\sessions.csv")

    def save(self, model: Model, results, input = None, activations = None) -> dict:
        exception = None
        try:
            self.model = model
            self.results = results
            self.input = input if input != None and type(input) == list else "not saved"
            self.activations = activations if activations != None and type(activations) == list else "not saved"
            payload = ""
            if not os.path.exists(self.path):
                payload = "\"session_id\",\"model_id\",\"processed_at\",\"results\",\"input\",\"activations\""
            with open(self.path, "a") as f:
                payload += f"\n\"{self.session_id}\",\"{self.model.id}\",\"{self.created_at}\",\"{self.results}\",\"{self.input}\",\"{self.activations}\""
                f.write(payload)
        except Exception as e:
            exception = e
        return {
            "status": True if exception is None else False,
            "session": self,
            "exception": exception
        }