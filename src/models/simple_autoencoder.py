import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_simple_autoencoder(input_shape=(999, 999, 1), encoding_dim=100):
    inputs = keras.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    encoded = layers.Dense(encoding_dim, activation='relu')(x)
    
    # Decoder
    x = layers.Dense(8 * (input_shape[0]//4) * (input_shape[1]//4), activation='relu')(encoded)
    x = layers.Reshape((input_shape[0]//4, input_shape[1]//4, 8))(x)
    x = layers.Conv2DTranspose(8, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(16, (3, 3), strides=2, activation='relu', padding='same')(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = keras.Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder
    
if __name__ == "__main__":
    model = build_simple_autoencoder()
    model.summary()
