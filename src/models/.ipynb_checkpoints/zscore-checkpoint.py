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
