#from CLI import CLI
from data_manager import Manager
from paramaters.weights import Weights
from objects.model import Model

if __name__ == "__main__":
    #data = Manager.load_data()
    Weights.from_dimensions([784, 16, 16, 10], "0", False)

    #model = Model.from_dimensions([784, 16, 16, 10])
    #model.fit(data) 
    
    #training_data, _, _ = Loader.load()
    #model = Model.from_dimensions([16, 16])
    #result = model.__forward([training_data[0][0]], [])
    #print(result)
    

    #training_data, _, _ = Loader.load()
    #model = Model.from_dimensions([16, 16])
    #result = model.infer(training_data[0][0])

    #CLI input will be substituted later when GUI is incorporated
    #CLI.main()