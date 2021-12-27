from lib.data_loader import FashionMNISTLoader
from lib.autoencoder import SimpleAutoencoder

from lib.plotter import plot_original_vs_generated

def main():
    loader = FashionMNISTLoader()
    x_train, x_test = loader.fit()


    autoencoder = SimpleAutoencoder()
    autoencoder.fit(x = x_train, y = x_test, epochs = 100)

    predictions = autoencoder.predict(x_test)

    original_shape = (x_test.shape[0], 28, 28)

    x_test = x_test.reshape(original_shape)
    predictions = predictions.reshape(original_shape)

    print(x_test.shape)
    print(predictions.shape)

    plot_original_vs_generated(x_test, predictions)

if __name__ == "__main__":
    main()
