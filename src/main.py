#from CLI import CLI
from model import Model
from data_loader import Loader

if __name__ == "__main__":
    training_data, _, _ = Loader.load_data()
    model = Model.from_dimensions([16, 16])
    session = model.infer(training_data[0][0])

    #CLI input will be substituted later when GUI is incorporated
    #CLI.main()