from lib.data_loader import FashionMNISTLoader


def main():
    loader = FashionMNISTLoader()
    x_train, x_test = loader.fit()

    print(x_train.shape)
    print(x_test.shape)

if __name__ == "__main__":
    main()
