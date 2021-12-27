from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Activation

def init_simple_autoencoder(input_shape, encoding_dim = 128):
    input_layer = Input(shape = (input_shape,))
    encoded = Dense(units = 512)(input_layer)
    encoded = ReLU()(encoded)
    encoded = Dense(units = 256)(encoded)
    encoded = ReLU()(encoded)

    encoded = Dense(encoding_dim)(encoded)
    encoding = ReLU()(encoded)

    decoded = Dense(units = 256)(encoding)
    decoded = ReLU()(decoded)
    decoded = Dense(units = 512)(decoded)
    encoding = ReLU()(decoded)

    decoded = Dense(units = input_shape)(decoded)
    decoded = Activation("sigmoid")(decoded)

    model = Model(input_layer, decoded)
    model.compile(optimizer = "adam", loss = "mse")

    return model
