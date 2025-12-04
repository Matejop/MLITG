from datetime import datetime as Date
import numpy as np
import hashlib
import orjson
import os

class FileManager():
    def save_model(model: dict) -> dict:
        exception = None
        try:
            dir_path = os.path.join(os.path.dirname(__file__), "..\\..\\models")
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            model["path"] = f"{dir_path}\\{model["name"]}_{model["created_at"]}.json"
            with open(model["path"], "w") as f:
                f.write(orjson.dumps(
                    model, 
                    default=lambda obj: obj.tolist() if type(obj) == np.ndarray else obj,  
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
            model = orjson.loads(open(path).read())
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
                result = FileManager.load_model(DirEntry)
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
    
    def save_session(results: np.array, model: dict) -> dict:
        session_id = None
        exception = None
        try:
            time_now = Date.now() 
            session_id = hashlib.md5(f"{model["id"]}+{time_now}".encode()).hexdigest()
            dir_path = os.path.join(os.path.dirname(__file__), "..\\..\\sessions.csv")
            payload = ""
            if not os.path.exists(dir_path):
               payload = "\"session_id\",\"model_id\",\"model_name\",\"results\",\"processed_at\"\n"
            with open(dir_path, "a") as f:
                f.write(payload + f"\"{session_id}\",\"{model["id"]}\",\"{model["name"] if model['name'] != "" else "\"\""}\",\"{results}\",\"{time_now}\"\n")
        except Exception as e:
            exception = e
        return {
            "status": True if exception is None else False,
            "session_id": session_id if exception is None else None,
            "exception": exception
        }
FileManager.load_models()