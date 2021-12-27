from lib.data_loader import FashionMNISTLoader
from lib.autoencoder import SimpleAutoencoder


def main():
    loader = FashionMNISTLoader()
    x_train, x_test = loader.fit()

    print(x_train.shape)
    print(x_test.shape)

    autoencoder = SimpleAutoencoder()
    autoencoder.fit(x = x_train, y = x_test)

if __name__ == "__main__":
    main()
