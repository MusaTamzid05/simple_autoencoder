from tensorflow.keras.datasets import fashion_mnist
import pickle

class DataLoader:
    def __init__(self):
        pass

    def process(self, x_data):
        raise RuntimeError("Need to be implemented")

    def fit(self):
        raise RuntimeError("Need to be implemented")




class FashionMNISTLoader(DataLoader):
    def __init__(self):
        super().__init__()
        (self.x_train, _), (self.x_test, _) = fashion_mnist.load_data()


    def process(self, x_data):
        x_data = x_data.astype("float32") / 255.0
        x_data = x_data.reshape(x_data.shape[0], -1)

        return x_data


    def fit(self):
        self.x_train = self.process(x_data = self.x_train)
        self.x_test = self.process(x_data = self.x_test)

        return self.x_train, self.x_test





class CatDogDataLoader(DataLoader):
    def __init__(self):
        super().__init__()

        with open("./custom_data/data.pickle", "rb") as f:
            data = pickle.load(f)
            self.x_train = data["train"]
            self.x_test = data["test"]



    def process(self, x_data):
        x_data = x_data.astype("float32") / 255.0
        x_data = x_data.reshape(x_data.shape[0], -1)

        return x_data


    def fit(self):
        self.x_train = self.process(x_data = self.x_train)
        self.x_test = self.process(x_data = self.x_test)

        return self.x_train, self.x_test





