from data_manager import DataManager
from objects.model import Model

if __name__ == "__main__":
    data = DataManager.load_data()
    training, validation, testing = data
    #model = Model.fit(data, [784, 64, 10])
    model = Model()
    model.load("C:\\Users\\matej\\source\\MLITG\\src\\objects\\../../models\\94c25e0e32140178575781e497016225.json")
    print(model.infer_dataset(testing))