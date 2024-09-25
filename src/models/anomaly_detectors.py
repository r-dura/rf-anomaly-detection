import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ZScoreDetector:
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.mean = None
        self.std = None
    
    def fit(self, X):
        X_reshaped = X.reshape(X.shape[0], -1)
        self.mean = np.mean(X_reshaped, axis=0)
        self.std = np.std(X_reshaped, axis=0)
    
    def predict(self, X):
        X_reshaped = X.reshape(X.shape[0], -1)
        z_scores = np.abs((X_reshaped - self.mean) / self.std)
        return np.max(z_scores, axis=1) > self.threshold

class AutoencoderDetector:
    def __init__(self, input_shape, threshold=0.1):
        self.threshold = threshold
        self.model = self.build_autoencoder(input_shape)
    
    def build_autoencoder(self, input_shape):
        input_layer = keras.Input(shape=input_shape)
        
        # Encoder
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
        encoded = layers.MaxPooling2D((2, 2), padding="same")(x)

        # Decoder
        x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(encoded)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(input_shape[-1], (3, 3), activation="linear", padding="same")(x)

        autoencoder = keras.Model(input_layer, decoded)
        autoencoder.compile(optimizer="adam", loss="mse")
        return autoencoder
    
    def fit(self, X, epochs=50, batch_size=32):
        self.model.fit(X, X, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    
    def predict(self, X):
        reconstructions = self.model.predict(X)
        mse = np.mean(np.square(X - reconstructions), axis=(1,2,3))
        return mse > self.threshold