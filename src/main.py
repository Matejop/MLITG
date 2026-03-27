from data_manager import DataManager
from objects.model import Model

if __name__ == "__main__":
    data = DataManager.load_data()
    model, testing_result = Model.fit(data, 784, 10)