from lib.models import init_simple_autoencoder

class Autoencoder:
    def __init__(self):
        self.model = None

    def load(self, model_dir_path):
        raise RuntimeError("Need to be implemented")

    def fit(self, x, y, epochs = 10, batch_size = 1024):
        raise RuntimeError("Need to be implemented")


    def save(self, model_dir_path):
        raise RuntimeError("Need to be implemented")


class SimpleAutoencoder(Autoencoder):
    def __init__(self):
        super().__init__()

    def load(self, model_dir_path):
        raise RuntimeError("Need to be implemented")

    def fit(self, x, y, epochs = 10, batch_size = 1024):
        print(f"Y shape :{y.shape}")

        if self.model is None:
            self.model = init_simple_autoencoder(input_shape = x.shape[1], encoding_dim = 128)
            self.model.summary()

        self.model.fit(
                x,
                x,
                epochs = epochs,
                batch_size = batch_size,
                shuffle = True,
                validation_data = (y, y)
                )

    def predict(self, test_data):
        return self.model.predict(test_data)






    def save(self, model_dir_path):
        raise RuntimeError("Need to be implemented")


