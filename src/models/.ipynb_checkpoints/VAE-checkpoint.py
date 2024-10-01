from tensorflow import keras
from tensorflow.keras import layers

def build_complex_autoencoder(input_shape=(999, 999), encoding_dim=100):
    inputs = keras.Input(shape=input_shape)
    x = layers.Reshape((999, 999, 1))(inputs)
    
    # Encoder
    x = layers.Conv2D(filters=8, kernel_size=(8,8), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(3)(x)
    x = layers.Dropout(.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=8, kernel_size=(4,4), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(3)(x)
    x = layers.Dropout(.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=8, kernel_size=(2,2), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(3)(x)
    x = layers.Dropout(.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=8, kernel_size=(9,9), padding="valid", activation="relu")(x)
    x = layers.GlobalMaxPool2D()(x)
    x = layers.Reshape((1,8))(x)

    encoded = layers.Dense(encoding_dim, activation="tanh", name="encoding")(x)
    
    # Decoder
    x = CellwiseKLDRegularizer()(encoded)
    x = layers.Dense(8)(x)
    x = layers.Reshape((1,1,8))(x)
    x = layers.UpSampling2D(29)(x)
    x = layers.Conv2DTranspose(filters=8, kernel_size=(9,9), padding="valid", activation="relu")(x)
    x = layers.Dropout(.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(3)(x)
    x = layers.Conv2DTranspose(filters=8, kernel_size=(4,4), padding="same", activation="relu")(x)
    x = layers.Dropout(.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(3)(x)
    x = layers.Conv2DTranspose(filters=8, kernel_size=(4,4), padding="same", activation="relu")(x)
    x = layers.Dropout(.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(3)(x)
    decoded = layers.Conv2DTranspose(filters=1, kernel_size=(8,8), padding="same", activation="linear")(x)

    autoencoder = keras.Model(inputs, decoded)
    return autoencoder

if __name__ == "__main__":
    model = build_complex_autoencoder()
    model.summary()
