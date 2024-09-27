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
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
    
    def predict(self, X):
        z_scores = np.abs((X - self.mean) / self.std)
        return np.any(z_scores > self.threshold, axis=1).astype(int)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ConvAutoencoderDetector:
    def __init__(self, input_shape, threshold=0.1):
        self.threshold = threshold
        self.input_shape = input_shape
        self.model = self.build_autoencoder()
    
    def build_autoencoder(self):
        input_layer = keras.Input(shape=self.input_shape)
        
        # Encoder
        x = layers.Conv1D(32, 3, activation="relu", padding="same")(input_layer)
        x = layers.MaxPooling1D(2, padding="same")(x)
        x = layers.Conv1D(16, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling1D(2, padding="same")(x)
        x = layers.Conv1D(8, 3, activation="relu", padding="same")(x)
        encoded = layers.MaxPooling1D(2, padding="same")(x)

        # Decoder
        x = layers.Conv1D(8, 3, activation="relu", padding="same")(encoded)
        x = layers.UpSampling1D(2)(x)
        x = layers.Conv1D(16, 3, activation="relu", padding="same")(x)
        x = layers.UpSampling1D(2)(x)
        x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
        x = layers.UpSampling1D(2)(x)
        decoded = layers.Conv1D(1, 3, activation="linear", padding="same")(x)
        
        autoencoder = keras.Model(input_layer, decoded)
        autoencoder.compile(optimizer="adam", loss="mse")
        return autoencoder
    
    def fit(self, X, epochs=50, batch_size=32):
        # Reshape X to (samples, timesteps, features)
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        self.model.fit(X_reshaped, X_reshaped, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    
    def predict(self, X):
        # Reshape X to (samples, timesteps, features)
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        reconstructions = self.model.predict(X_reshaped)
        mse = np.mean(np.square(X_reshaped - reconstructions), axis=(1,2))
        return mse > self.threshold